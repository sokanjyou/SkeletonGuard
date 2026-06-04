from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any


def build_logger(name: str = "fall3d") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)


@dataclass(frozen=True)
class MQTTAlertConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    topic: str = "fall3d/alerts"
    client_id: str = "fall3d-detector"
    username: str | None = None
    password: str | None = None
    qos: int = 1
    retain: bool = False
    keepalive: int = 60
    connect_timeout_s: float = 3.0
    min_interval_s: float = 2.0
    extra: dict[str, Any] = field(default_factory=dict)


class MQTTAlertPublisher:
    def __init__(self, config: MQTTAlertConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.client = None
        self.connected = False
        self._last_publish_by_key: dict[str, float] = {}

    def start(self) -> None:
        if not self.config.enabled:
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.logger.warning("MQTT alerts are enabled but paho-mqtt is not installed.")
            return

        client = mqtt.Client(client_id=self.config.client_id)
        if self.config.username:
            client.username_pw_set(self.config.username, self.config.password)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        try:
            client.connect(
                self.config.host,
                int(self.config.port),
                keepalive=int(self.config.keepalive),
            )
        except OSError as exc:
            self.logger.warning("MQTT alert publisher could not connect: %s", exc)
            return
        client.loop_start()
        self.client = client

        deadline = time.time() + float(self.config.connect_timeout_s)
        while not self.connected and time.time() < deadline:
            time.sleep(0.05)
        if self.connected:
            self.logger.info(
                "MQTT alert publisher connected to %s:%s topic=%s",
                self.config.host,
                self.config.port,
                self.config.topic,
            )
        else:
            self.logger.warning(
                "MQTT alert publisher did not confirm connection within %.1fs.",
                self.config.connect_timeout_s,
            )

    def publish(self, message: str, payload: dict[str, Any] | None = None) -> None:
        if not self.config.enabled or self.client is None:
            return
        key = self._rate_limit_key(message, payload)
        now = time.time()
        last_publish = self._last_publish_by_key.get(key, 0.0)
        if now - last_publish < max(0.0, float(self.config.min_interval_s)):
            return
        self._last_publish_by_key[key] = now
        data = {
            "event": "fall_alert",
            "timestamp": now,
            "message": message,
            **self.config.extra,
        }
        if payload:
            data.update(payload)
        result = self.client.publish(
            self.config.topic,
            json.dumps(data, ensure_ascii=False),
            qos=int(self.config.qos),
            retain=bool(self.config.retain),
        )
        if result.rc != 0:
            self.logger.warning("MQTT alert publish failed with rc=%s.", result.rc)

    def stop(self) -> None:
        if self.client is None:
            return
        self.client.loop_stop()
        self.client.disconnect()
        self.client = None
        self.connected = False
        self._last_publish_by_key.clear()

    def _on_connect(self, client, userdata, flags, rc, *args):
        del client, userdata, flags, args
        self.connected = rc == 0
        if rc != 0:
            self.logger.warning("MQTT alert publisher connect failed with rc=%s.", rc)

    def _on_disconnect(self, client, userdata, rc, *args):
        del client, userdata, args
        self.connected = False
        if rc != 0:
            self.logger.warning("MQTT alert publisher disconnected with rc=%s.", rc)

    @staticmethod
    def _rate_limit_key(message: str, payload: dict[str, Any] | None) -> str:
        if payload and payload.get("track_id") is not None:
            return f"track:{payload['track_id']}"
        return message


def build_mqtt_alert_publisher(
    cfg: dict[str, Any] | None,
    logger: logging.Logger,
) -> MQTTAlertPublisher | None:
    cfg = cfg or {}
    mqtt_cfg = cfg.get("mqtt") or cfg
    config = MQTTAlertConfig(**mqtt_cfg)
    if not config.enabled:
        return None
    publisher = MQTTAlertPublisher(config, logger)
    publisher.start()
    return publisher


def emit_alert(
    logger: logging.Logger,
    message: str,
    publisher: MQTTAlertPublisher | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    logger.warning("FALL_ALERT %s", message)
    if publisher is not None:
        publisher.publish(message, payload=payload)
