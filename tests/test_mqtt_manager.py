"""Tests for MqttManager orchestration."""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import pytest

from bridge.mqtt_manager import MqttManager
from bridge.message_parser import parse_and_publish
from bridge.mqtt_publish import safe_publish
from tests.fakes import FakeBrokerClient, FakeSerialConnection, make_test_state, make_config


class StartupBrokerClient(FakeBrokerClient):
    """Drive connection callbacks without a network or background threads."""

    def __init__(self, manager, broker_idx):
        super().__init__()
        self.manager = manager
        self.broker_idx = broker_idx
        self.loop_starts = 0

    def connect(self, server, port, keepalive=60):
        self.connect_calls.append((server, port, keepalive))
        if self.broker_idx in self.manager.blocked:
            self.manager.connect_started.set()
            if not self.manager.allow_connect.wait(timeout=5):
                raise TimeoutError('Test did not release connection attempt')
        if self.broker_idx in self.manager.unreachable:
            raise OSError(113, 'No route to host')
        self._connected = True

    def loop_start(self):
        self.loop_starts += 1
        self.manager.on_mqtt_connect(
            self, {'name': str(self.broker_idx), 'broker_idx': self.broker_idx}, None, 0,
        )


class StartupMqttManager(MqttManager):
    def __init__(self, state, unreachable):
        super().__init__(state)
        self.unreachable = set(unreachable)
        self.blocked = set()
        self.connect_started = threading.Event()
        self.allow_connect = threading.Event()
        self.created = []

    def _create_broker_client(self, broker_idx):
        client = StartupBrokerClient(self, broker_idx)
        self.created.append(client)
        return client


class TestInitialConnectionRecovery:
    def make_manager(self, unreachable):
        config = make_config(broker=[
            {'name': 'first', 'enabled': True, 'server': 'first'},
            {'name': 'disabled', 'enabled': False, 'server': 'disabled'},
            {'name': 'last', 'enabled': True, 'server': 'last'},
        ])
        state = make_test_state(config=config, repeater_name='Node', repeater_pub_key='AA' * 32)
        return state, StartupMqttManager(state, unreachable)

    @pytest.mark.parametrize('failed_idx', [0, 2])
    def test_partial_startup_failure_recovers(self, failed_idx):
        state, manager = self.make_manager({failed_idx})
        before = time.time()
        assert manager.connect_all_brokers() is True
        after = time.time()
        assert [info['broker_idx'] for info in state.mqtt_clients] == [0, 2]
        failed = next(info for info in state.mqtt_clients if info['broker_idx'] == failed_idx)
        healthy = next(info for info in state.mqtt_clients if info['broker_idx'] != failed_idx)
        healthy_client = healthy['client']
        assert failed['client'] is None
        assert failed['connected'] is False
        assert failed['connecting_since'] == 0
        assert failed['failed_attempts'] == 1
        assert before + 0.5 <= failed['reconnect_at'] <= after + 1.5
        assert manager.created[failed_idx // 2].disconnect_calls == 1

        # A pending entry must not cause publish failures on healthy brokers.
        assert safe_publish(state, 'debug', 'packet') is True
        assert state.stats['publish_failures'] == 0

        failed['reconnect_at'] = time.time() + 60
        manager.reconnect_disconnected_brokers()
        assert len(manager.created) == 2

        manager.unreachable.clear()
        failed['reconnect_at'] = 0
        manager.reconnect_disconnected_brokers()
        assert all(info['connected'] for info in state.mqtt_clients)
        recovered = next(info for info in state.mqtt_clients if info['broker_idx'] == failed_idx)
        assert recovered['client'].connect_calls == [(failed['server'], 1883, 60)]
        assert recovered['client'].loop_starts == 1
        assert healthy['client'] is healthy_client
        assert healthy_client.disconnect_calls == 0
        assert len(manager.created) == 3

    def test_all_initial_failures_reuse_entries_and_recover(self):
        state, manager = self.make_manager({0, 2})
        assert manager.connect_all_brokers() is False
        assert len(state.mqtt_clients) == 2
        manager.unreachable.clear()
        for info in state.mqtt_clients:
            info['reconnect_at'] = 0
        assert manager.connect_all_brokers() is True
        assert manager.connect_all_brokers() is True
        assert [info['broker_idx'] for info in state.mqtt_clients] == [0, 2]
        assert len(manager.created) == 4
        assert all(info['client'].loop_starts == 1 for info in state.mqtt_clients)

    def test_persistent_failure_keeps_healthy_broker_and_eventually_recovers(self):
        state, manager = self.make_manager({0})
        assert manager.connect_all_brokers() is True
        failed = state.mqtt_clients[0]
        delay = failed['reconnect_delay']
        failed['reconnect_at'] = 0
        before = time.time()
        manager.reconnect_disconnected_brokers()
        assert failed['failed_attempts'] == 2
        assert failed['reconnect_delay'] == delay * state.reconnect_backoff
        assert failed['reconnect_at'] >= before + delay - 0.5
        healthy_client = state.mqtt_clients[1]['client']
        # Exceed the former 12-failure restart threshold and reach the delay cap.
        for _ in range(25):
            failed['reconnect_at'] = 0
            manager.reconnect_disconnected_brokers()
        assert failed['failed_attempts'] == 27
        assert failed['reconnect_delay'] == state.max_reconnect_delay
        assert state.should_exit is False
        assert state.mqtt_connected is True
        assert safe_publish(state, 'debug', 'still publishing') is True
        assert healthy_client.disconnect_calls == 0

        manager.unreachable.clear()
        failed['reconnect_at'] = 0
        manager.reconnect_disconnected_brokers()
        assert all(info['connected'] for info in state.mqtt_clients)
        assert state.mqtt_clients[1]['client'] is healthy_client

    def test_background_retry_does_not_block_serial_reading_and_publishing(self):
        state, manager = self.make_manager({0})
        assert manager.connect_all_brokers() is True
        manager.unreachable.clear()
        manager.blocked.add(0)
        state.mqtt_clients[0]['reconnect_at'] = 0
        healthy_client = state.mqtt_clients[1]['client']
        state.device = FakeSerialConnection(lines=[
            '12:34:56 - 1/15/2025 U: TX, len=32 (type=2, route=F, payload_len=16)',
        ])
        try:
            manager.start_reconnect_loop()
            worker = manager._retry_thread
            manager.start_reconnect_loop()
            assert manager._retry_thread is worker
            assert manager.connect_started.wait(timeout=2)

            # The network attempt stays blocked while the reader parses and
            # publishes a real packet through the healthy connection.
            parse_and_publish(state, state.device.read_line())
            packet = json.loads(healthy_client.published[-1][1])
            assert packet['type'] == 'PACKET'
            assert packet['direction'] == 'tx'
            assert state.stats['packets_tx'] == 1
            assert manager.allow_connect.is_set() is False
            assert healthy_client.disconnect_calls == 0
        finally:
            manager.allow_connect.set()
            manager.stop_reconnect_loop(timeout=2)
        assert worker.is_alive() is False

    def test_shutdown_discards_connection_that_finishes_after_worker_stop(self):
        state, manager = self.make_manager({0})
        assert manager.connect_all_brokers() is True
        pending = state.mqtt_clients[0]
        pending['reconnect_at'] = 0
        manager.unreachable.clear()
        manager.blocked.add(0)
        try:
            manager.start_reconnect_loop()
            assert manager.connect_started.wait(timeout=2)
            # Shutdown must return even while connect() is still blocked.
            manager.stop_reconnect_loop(timeout=0)
            assert manager._retry_thread.is_alive() is True
        finally:
            manager.allow_connect.set()
            manager.stop_reconnect_loop(timeout=2)
        assert manager._retry_thread.is_alive() is False
        assert state.mqtt_clients[0] is pending
        late_client = manager.created[-1]
        assert late_client.loop_starts == 0
        assert late_client.disconnect_calls == 1
        manager.start_reconnect_loop()
        assert manager._retry_thread.is_alive() is False

    def test_disabled_brokers_are_not_registered(self):
        state, manager = self.make_manager(set())
        for broker in state.config['broker']:
            broker['enabled'] = False
        assert manager.connect_all_brokers() is False
        assert state.mqtt_clients == []
        assert manager.created == []


class TestMqttManagerCallbacks:
    def _make_manager(self, **state_kwargs):
        state = make_test_state(
            repeater_name="TestNode",
            repeater_pub_key="AA" * 32,
            **state_kwargs,
        )
        manager = MqttManager(state)
        state.mqtt_manager = manager
        return state, manager

    def test_on_connect_sets_connected(self):
        state, manager = self._make_manager()
        broker = FakeBrokerClient()
        broker._connected = True
        state.mqtt_clients = [{"client": broker, "broker_idx": 0, "connected": False, "connecting_since": 1, "connect_time": 0, "failed_attempts": 0}]

        import threading
        state.connection_events[0] = threading.Event()

        manager.on_mqtt_connect(None, {'name': 'test', 'broker_idx': 0}, None, 0)

        assert state.mqtt_clients[0]['connected'] is True
        assert state.mqtt_connected is True

    def test_on_connect_rejects_nonzero_rc(self):
        state, manager = self._make_manager()
        import threading
        state.connection_events[0] = threading.Event()

        manager.on_mqtt_connect(None, {'name': 'test', 'broker_idx': 0}, None, 1)
        assert state.mqtt_connected is False

    def test_on_disconnect_marks_disconnected(self):
        state, manager = self._make_manager()
        state.mqtt_clients = [{"client": FakeBrokerClient(), "broker_idx": 0, "connected": True, "connecting_since": 0, "connect_time": 100, "reconnect_at": 0, "failed_attempts": 0}]
        state.mqtt_connected = True

        manager.on_mqtt_disconnect(None, {'name': 'test', 'broker_idx': 0}, None, 0, None)

        assert state.mqtt_clients[0]['connected'] is False
        assert state.mqtt_connected is False

    def test_on_message_ignores_non_serial(self):
        state, manager = self._make_manager()
        msg = type('Msg', (), {'topic': 'other/topic', 'payload': b'test'})()
        # Should not raise
        manager.on_mqtt_message(None, {'name': 'test', 'broker_idx': 0}, msg)


class TestReconnectBehavior:
    def test_skips_connected_brokers(self):
        state = make_test_state(repeater_name="TestNode", repeater_pub_key="AA" * 32)
        manager = MqttManager(state)
        state.mqtt_manager = manager

        broker = FakeBrokerClient()
        state.mqtt_clients = [{"client": broker, "broker_idx": 0, "connected": True, "connecting_since": 0, "connect_time": 100, "reconnect_at": 0, "failed_attempts": 0}]

        # Should not attempt reconnect for connected broker
        manager.reconnect_disconnected_brokers()
        assert state.mqtt_clients[0]['connected'] is True

    def test_clears_token_cache_on_reconnect(self):
        config = make_config()
        state = make_test_state(
            config=config,
            repeater_name="TestNode",
            repeater_pub_key="AA" * 32,
        )
        manager = MqttManager(state)
        state.mqtt_manager = manager

        state.token_cache[0] = ("cached_token", 1000)

        broker = FakeBrokerClient()
        state.mqtt_clients = [{
            "client": broker,
            "broker_idx": 0,
            "connected": False,
            "connecting_since": 0,
            "connect_time": 100,
            "reconnect_at": 0,
            "reconnect_delay": 1.0,
            "failed_attempts": 0,
        }]

        # Reconnect will try to create a new client, which may fail since
        # there's no real MQTT server, but it should clear the token cache
        manager.reconnect_disconnected_brokers()
        assert 0 not in state.token_cache

    def test_backoff_is_independent_per_broker(self):
        """Two brokers maintain separate reconnect_delay counters.

        Broker 0 fails repeatedly so its backoff grows.
        Broker 1 stays connected throughout, then disconnects.
        Broker 1's reconnect_delay must still be 1.0 — unaffected by broker 0.
        When broker 1 subsequently fails its first reconnect attempt its own
        backoff starts at 1.0 and grows independently from broker 0's.
        """
        config = make_config()
        config['broker'] = [
            {'name': 'b0', 'enabled': True, 'server': 'host0', 'port': 1883,
             'transport': 'tcp', 'qos': 0, 'retain': False,
             'auth': {'method': 'none'}, 'tls': {'enabled': False}},
            {'name': 'b1', 'enabled': True, 'server': 'host1', 'port': 1883,
             'transport': 'tcp', 'qos': 0, 'retain': False,
             'auth': {'method': 'none'}, 'tls': {'enabled': False}},
        ]
        state = make_test_state(config=config, repeater_name="Node", repeater_pub_key="AA" * 32)
        manager = MqttManager(state)
        state.mqtt_manager = manager

        def make_client_info(idx, connected):
            return {
                'client': FakeBrokerClient(),
                'broker_idx': idx,
                'connected': connected,
                'connecting_since': 0,
                'connect_time': time.time() - 300 if connected else 0,
                'reconnect_at': 0,
                'reconnect_delay': 1.0,
                'failed_attempts': 0,
            }

        state.mqtt_clients = [make_client_info(0, False), make_client_info(1, True)]
        state.mqtt_connected = True

        # Fail broker 0 five times — its delay should grow: 1.0 → 1.5 → 2.25 → 3.375 → 5.06 → 7.59
        with patch.object(manager, '_create_and_connect_broker', return_value=None):
            for _ in range(5):
                state.mqtt_clients[0]['reconnect_at'] = 0
                manager.reconnect_disconnected_brokers()

        assert state.mqtt_clients[0]['reconnect_delay'] == pytest.approx(1.0 * 1.5 ** 5, rel=1e-3)
        assert state.mqtt_clients[1]['reconnect_delay'] == 1.0, \
            "broker 1 delay must be unaffected by broker 0 failures"

        # Broker 1 disconnects — reconnect_at should use its own delay (1.0)
        t_before = time.time()
        manager.on_mqtt_disconnect(None, {'name': 'b1', 'broker_idx': 1}, None, 0, None)
        t_after = time.time()

        assert state.mqtt_clients[1]['reconnect_delay'] == 1.0
        assert t_before + 1.0 <= state.mqtt_clients[1]['reconnect_at'] <= t_after + 1.0

        # Broker 1 fails its first reconnect — its own backoff starts from 1.0
        with patch.object(manager, '_create_and_connect_broker', return_value=None):
            state.mqtt_clients[1]['reconnect_at'] = 0
            manager.reconnect_disconnected_brokers()

        assert state.mqtt_clients[1]['reconnect_delay'] == pytest.approx(1.5, rel=1e-3), \
            "broker 1 first failure must grow from 1.0, not from broker 0's grown value"
