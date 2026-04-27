"""
Navigator - Runtime Status Publisher

Writes the current robot state atomically to a JSON file and optionally
broadcasts it over WebSocket to connected screen clients.

Design constraints:
- Never blocks or crashes the audio runtime
- JSON write is atomic (write tmp + rename)
- WebSocket is fully optional; missing library or network errors are logged and swallowed
- Thread-safe: publish() may be called from any thread
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


class StatusPublisher:
    """
    Publish robot runtime state to a JSON file and (optionally) WebSocket clients.

    Args:
        json_path:  File path for the status JSON (created/replaced atomically).
        ws_enabled: Whether to start a WebSocket server.
        ws_host:    WebSocket bind host.
        ws_port:    WebSocket bind port.
    """

    def __init__(
        self,
        json_path: str = "data/runtime_status.json",
        ws_enabled: bool = False,
        ws_host: str = "127.0.0.1",
        ws_port: int = 8765,
    ) -> None:
        self._json_path = Path(json_path)
        self._ws_enabled = ws_enabled
        self._ws_host = ws_host
        self._ws_port = ws_port

        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._ws_clients: set[Any] = set()
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create the JSON directory and start the WebSocket server if enabled."""
        try:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("status_publisher_mkdir_failed", error=str(exc))

        if self._ws_enabled:
            self._start_ws_server()

    def stop(self) -> None:
        """Shut down the WebSocket server gracefully."""
        loop = self._ws_loop
        if loop and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
        thread = self._ws_thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish(
        self,
        event: str,
        state: str,
        message: str,
        session_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Publish a status update.

        Keyword args in ``extra`` override the auto-derived boolean fields
        (is_listening, is_speaking, wake_word_detected).
        """
        payload: dict[str, Any] = {
            "event": event,
            "state": state,
            "message": message,
            "is_listening": extra.pop("is_listening", state == "listening"),
            "is_speaking": extra.pop("is_speaking", state == "speaking"),
            "wake_word_detected": extra.pop("wake_word_detected", state not in ("idle", "error")),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        payload.update(extra)

        with self._lock:
            self._latest = payload

        self._write_json_safe(payload)
        if self._ws_enabled:
            self._broadcast_ws_safe(payload)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _write_json_safe(self, payload: dict[str, Any]) -> None:
        try:
            tmp = self._json_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._json_path)
        except Exception as exc:
            logger.warning("status_publisher_write_failed", error=str(exc))

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    def _start_ws_server(self) -> None:
        try:
            import websockets  # type: ignore  # noqa: F401
        except ImportError:
            logger.warning("status_ws_disabled", reason="websockets_not_installed")
            return

        def _thread_main() -> None:
            loop = asyncio.new_event_loop()
            self._ws_loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_ws_server())
            except Exception as exc:
                logger.warning("status_ws_server_error", error=str(exc))
            finally:
                loop.close()

        self._ws_thread = threading.Thread(
            target=_thread_main,
            daemon=True,
            name="status-ws-server",
        )
        self._ws_thread.start()
        logger.info("status_ws_server_started", host=self._ws_host, port=self._ws_port)

    async def _run_ws_server(self) -> None:
        import websockets  # type: ignore

        async def _handler(ws: Any) -> None:
            # Send latest status immediately on connect
            with self._lock:
                latest = dict(self._latest)
            if latest:
                try:
                    await ws.send(json.dumps(latest, ensure_ascii=False))
                except Exception:
                    return

            with self._lock:
                self._ws_clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                with self._lock:
                    self._ws_clients.discard(ws)

        try:
            server = await websockets.serve(_handler, self._ws_host, self._ws_port)
            await server.wait_closed()
        except Exception as exc:
            logger.warning("status_ws_bind_failed", host=self._ws_host, port=self._ws_port, error=str(exc))

    def _broadcast_ws_safe(self, payload: dict[str, Any]) -> None:
        loop = self._ws_loop
        if not loop or loop.is_closed():
            return

        msg = json.dumps(payload, ensure_ascii=False)

        async def _send_all() -> None:
            with self._lock:
                clients = set(self._ws_clients)
            dead: set[Any] = set()
            for ws in clients:
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            if dead:
                with self._lock:
                    self._ws_clients -= dead

        coro = _send_all()
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:
            coro.close()  # prevent "coroutine never awaited" warning
            logger.warning("status_ws_broadcast_failed", error=str(exc))
