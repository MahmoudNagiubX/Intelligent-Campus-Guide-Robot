"""
Navigator - MQTT State Publisher

Publishes a single short state string to an MQTT broker whenever the robot
state changes.  Strings are exactly:

    "wait"   — robot is idle, waiting for the wake word
    "listen" — active session, STT/Deepgram is live
    "speak"  — TTS playback is active

Design constraints:
- Never blocks or crashes the audio runtime
- Skips publishing when the same state is published twice in a row
- paho-mqtt loop_start() provides a background network thread; publish() is safe
- If paho-mqtt is not installed or the broker is unreachable, logs a warning
  and continues silently
- MQTT disabled (mqtt_enabled=false) means no import, no connection, no-op
"""

from __future__ import annotations

import ssl
import threading
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Map internal status-publisher state strings → short MQTT strings
_STATE_MAP: dict[str, str] = {
    "idle": "wait",
    "error": "wait",
    "wake_detected": "listen",
    "listening": "listen",
    "processing": "listen",
    "speaking": "speak",
}


class MQTTPublisher:
    """
    Publishes simplified robot state to an MQTT broker.

    Args:
        enabled:     Whether MQTT publishing is active.
        broker:      Broker hostname (e.g. HiveMQ Cloud FQDN).
        port:        Broker port (8883 for TLS).
        username:    MQTT username.
        password:    MQTT password.
        topic:       Topic to publish to.
        tls_enabled: Connect with TLS.
        qos:         MQTT QoS level (0 / 1 / 2).
        retain:      Set the retain flag on published messages.
    """

    def __init__(
        self,
        enabled: bool = False,
        broker: str = "",
        port: int = 8883,
        username: str = "",
        password: str = "",
        topic: str = "inno/ai",
        tls_enabled: bool = True,
        qos: int = 1,
        retain: bool = True,
    ) -> None:
        self._enabled = enabled
        self._broker = broker
        self._port = port
        self._username = username
        self._password = password
        self._topic = topic
        self._tls_enabled = tls_enabled
        self._qos = qos
        self._retain = retain

        self._lock = threading.Lock()
        self._last_published: Optional[str] = None
        self._client = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to the broker and start the background network loop."""
        if not self._enabled or not self._broker:
            return
        self._connect_safe()

    def stop(self) -> None:
        """Publish a final "wait" and disconnect cleanly."""
        client = self._client
        if client is None:
            return
        try:
            client.publish(self._topic, "wait", qos=self._qos, retain=self._retain)
            client.loop_stop()
            client.disconnect()
        except Exception as exc:
            logger.warning("mqtt_stop_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish_state(self, state: str) -> None:
        """
        Map *state* to a short MQTT string and publish if it changed.

        Unknown states are silently ignored.
        """
        if not self._enabled:
            return
        msg = _STATE_MAP.get(state)
        if msg is None:
            return
        with self._lock:
            if msg == self._last_published:
                return
            self._last_published = msg
        self._publish_safe(msg)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect_safe(self) -> None:
        try:
            import paho.mqtt.client as mqtt  # type: ignore

            # paho-mqtt >= 2.0 requires explicit callback API version.
            try:
                client = mqtt.Client(
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                    protocol=mqtt.MQTTv5,
                )
            except AttributeError:
                # Fallback for paho-mqtt < 2.0
                client = mqtt.Client()

            if self._username:
                client.username_pw_set(self._username, self._password or None)

            if self._tls_enabled:
                client.tls_set(
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLS_CLIENT,
                )

            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect

            client.connect_async(self._broker, self._port, keepalive=60)
            client.loop_start()
            self._client = client

        except ImportError:
            logger.warning("mqtt_disabled", reason="paho-mqtt_not_installed")
        except Exception as exc:
            logger.warning("mqtt_connect_failed", broker=self._broker, port=self._port, error=str(exc))

    def _publish_safe(self, message: str) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.publish(self._topic, message, qos=self._qos, retain=self._retain)
            logger.debug("mqtt_published", topic=self._topic, message=message)
        except Exception as exc:
            logger.warning("mqtt_publish_failed", error=str(exc))

    # Callback signatures for paho-mqtt 2.x (VERSION2)
    def _on_connect(self, client, userdata, connect_flags, reason_code, properties=None) -> None:
        code = reason_code.value if hasattr(reason_code, "value") else reason_code
        if code == 0:
            logger.info(
                "mqtt_connected",
                broker=self._broker,
                port=self._port,
                topic=self._topic,
            )
        else:
            logger.warning("mqtt_connect_refused", reason_code=str(reason_code))

    def _on_disconnect(self, client, userdata, disconnect_flags=None, reason_code=None, properties=None) -> None:
        logger.info("mqtt_disconnected", reason_code=str(reason_code) if reason_code else "clean")
