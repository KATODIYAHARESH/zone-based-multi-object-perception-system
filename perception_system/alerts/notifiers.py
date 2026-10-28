"""Alert delivery channels and the asynchronous dispatcher.

The video loop only *enqueues*; a worker thread delivers with retries,
exponential back-off and per-channel token-bucket rate limits, so a slow SMTP
server or a Telegram outage can never stall perception.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import smtplib
import ssl
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, deque
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from core.types import Alert, Severity, ZoneEvent, iso_utc

log = logging.getLogger(__name__)


class _TokenBucket:
    def __init__(self, per_minute: int) -> None:
        self.capacity = float(max(1, per_minute))
        self.tokens = self.capacity
        self.rate = self.capacity / 60.0
        self.t = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class Notifier(ABC):
    name = "notifier"

    def __init__(self, min_severity: str = "info", max_per_minute: int = 60) -> None:
        self.min_severity = Severity.parse(min_severity)
        self.bucket = _TokenBucket(max_per_minute)

    def accepts(self, alert: Alert) -> bool:
        return alert.severity.rank >= self.min_severity.rank

    @abstractmethod
    def send(self, alert: Alert, snapshot: Optional[bytes]) -> None:
        """Deliver one alert; raise on failure (the dispatcher retries)."""


class OverlayNotifier(Notifier):
    """In-memory ring buffer read by the on-screen overlay and dashboards."""

    name = "overlay"

    def __init__(self, max_items: int = 50) -> None:
        super().__init__("info", 1_000_000)
        self._items: Deque[Alert] = deque(maxlen=max_items)
        self._lock = threading.Lock()

    def send(self, alert: Alert, snapshot: Optional[bytes]) -> None:
        with self._lock:
            self._items.appendleft(alert)

    def recent(self, n: int = 10) -> List[Alert]:
        with self._lock:
            return list(self._items)[:n]


class LogFileNotifier(Notifier):
    """Append-only JSON-lines alert log, rotated daily (UTC)."""

    name = "log_file"

    def __init__(self, path: Path, min_severity: str = "info") -> None:
        super().__init__(min_severity, 1_000_000)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"zbmops.alerts.{path}")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.handlers.TimedRotatingFileHandler(path, when="midnight", utc=True, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def send(self, alert: Alert, snapshot: Optional[bytes]) -> None:
        self._logger.info(json.dumps(alert.to_dict(), ensure_ascii=False))


class TelegramNotifier(Notifier):
    name = "telegram"
    _API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str, chat_id: str, send_snapshot: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        if not token or not chat_id:
            raise ValueError("Telegram bot token / chat id missing (set the configured environment variables)")
        self.token, self.chat_id, self.send_snapshot = token, chat_id, send_snapshot

    def send(self, alert: Alert, snapshot: Optional[bytes]) -> None:
        import requests

        text = f"[{alert.severity.value.upper()}] {alert.message}\n{iso_utc(alert.timestamp)} | rule {alert.rule_id}"
        if snapshot and self.send_snapshot:
            r = requests.post(self._API.format(token=self.token, method="sendPhoto"),
                              data={"chat_id": self.chat_id, "caption": text[:1024]},
                              files={"photo": ("alert.jpg", snapshot, "image/jpeg")}, timeout=10)
        else:
            r = requests.post(self._API.format(token=self.token, method="sendMessage"),
                              data={"chat_id": self.chat_id, "text": text}, timeout=10)
        r.raise_for_status()


class EmailNotifier(Notifier):
    name = "email"

    def __init__(self, host: str, port: int, username: str, password: str, sender: str,
                 recipients: Sequence[str], use_tls: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        if not host or not recipients:
            raise ValueError("SMTP host / recipients missing")
        self.host, self.port, self.username, self.password = host, port, username, password
        self.sender, self.recipients, self.use_tls = sender, list(recipients), use_tls

    def send(self, alert: Alert, snapshot: Optional[bytes]) -> None:
        msg = EmailMessage()
        msg["Subject"] = f"[ZBMOPS][{alert.severity.value.upper()}] {alert.message}"[:200]
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(json.dumps(alert.to_dict(), indent=2))
        if snapshot:
            msg.add_attachment(snapshot, maintype="image", subtype="jpeg", filename="alert.jpg")
        with smtplib.SMTP(self.host, self.port, timeout=15) as smtp:
            if self.use_tls:
                smtp.starttls(context=ssl.create_default_context())
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(msg)


def alert_to_web(alert: Alert) -> Dict[str, Any]:
    return {"type": "alert", "ts": iso_utc(alert.timestamp), "severity": alert.severity.value,
            "ruleId": alert.rule_id, "zoneId": alert.zone_id, "zoneName": alert.zone_name,
            "trackId": alert.track_id, "category": alert.category, "message": alert.message,
            "meta": {"rule_type": alert.rule_type, **{k: v for k, v in alert.data.items() if isinstance(v, (int, float, str))}}}


def zone_event_to_web(ev: ZoneEvent) -> Dict[str, Any]:
    x, y = ev.anchor_norm if ev.anchor_norm else (None, None)
    return {"type": ev.type.value, "ts": iso_utc(ev.timestamp), "zoneId": ev.zone_id, "zoneName": ev.zone_name,
            "trackId": ev.track_id, "category": ev.category.value, "label": ev.label,
            "dwellS": round(ev.dwell_s, 2), "x": x, "y": y}


class WebhookNotifier(Notifier):
    """POSTs alerts (and, via :meth:`post`, zone events / heartbeats) to the web console."""

    name = "webhook"

    def __init__(self, url: str, token: Optional[str], timeout_s: float = 3.0,
                 heartbeat_url: Optional[str] = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.url, self.token, self.timeout_s, self.heartbeat_url = url, token, timeout_s, heartbeat_url

    def post(self, payload: Dict[str, Any], url: Optional[str] = None) -> None:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        r = requests.post(url or self.url, data=json.dumps(payload), headers=headers, timeout=self.timeout_s)
        r.raise_for_status()

    def send(self, alert: Alert, snapshot: Optional[bytes]) -> None:
        self.post({"source": "engine", "events": [alert_to_web(alert)]})


class AlertDispatcher:
    """Single background worker that delivers alerts and generic network tasks."""

    def __init__(self, notifiers: Sequence[Notifier], queue_size: int = 512, retries: int = 3) -> None:
        self.notifiers = list(notifiers)
        self.retries = retries
        self._q: "queue.Queue[Tuple[str, Any, Any]]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.dropped = 0
        self.failures: Counter = Counter()
        self._thread = threading.Thread(target=self._run, name="alert-dispatch", daemon=True)
        self._thread.start()

    def _put(self, item: Tuple[str, Any, Any]) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:  # drop the OLDEST item: fresh information is worth more
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self.dropped += 1

    def dispatch(self, alert: Alert, snapshot: Optional[bytes] = None) -> None:
        self._put(("alert", alert, snapshot))

    def submit(self, task: Callable[[], None]) -> None:
        self._put(("task", task, None))

    def _deliver(self, n: Notifier, alert: Alert, snapshot: Optional[bytes]) -> None:
        if not n.accepts(alert):
            return
        if not n.bucket.take():
            log.warning("Rate limit on channel %s - alert %s not delivered", n.name, alert.rule_id)
            return
        for attempt in range(1, self.retries + 1):
            try:
                n.send(alert, snapshot)
                return
            except Exception as exc:
                if attempt == self.retries:
                    self.failures[n.name] += 1
                    log.error("Channel %s failed after %d attempts: %s", n.name, attempt, exc)
                else:
                    time.sleep(min(4.0, 0.5 * 2 ** (attempt - 1)))

    def _run(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                kind, payload, extra = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == "alert":
                for n in self.notifiers:
                    self._deliver(n, payload, extra)
            else:
                try:
                    payload()
                except Exception as exc:
                    self.failures["task"] += 1
                    log.debug("Background task failed: %s", exc)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)


def build_notifiers(cfg: Any, resolve: Callable[[str], Path]) -> Tuple[List[Notifier], Optional[OverlayNotifier], Optional[WebhookNotifier]]:
    """Create enabled channels from ``AlertsConfig``; misconfigured channels are logged, not fatal."""
    ch = cfg.channels
    notifiers: List[Notifier] = []
    overlay: Optional[OverlayNotifier] = None
    webhook: Optional[WebhookNotifier] = None
    if ch.overlay.enabled:
        overlay = OverlayNotifier(ch.overlay.max_items)
        notifiers.append(overlay)
    if ch.log_file.enabled:
        notifiers.append(LogFileNotifier(resolve(ch.log_file.path), ch.log_file.min_severity))
    if ch.telegram.enabled:
        try:
            notifiers.append(TelegramNotifier(os.environ.get(ch.telegram.bot_token_env, ""),
                                              os.environ.get(ch.telegram.chat_id_env, ""), ch.telegram.send_snapshot,
                                              min_severity=ch.telegram.min_severity, max_per_minute=ch.telegram.max_per_minute))
        except ValueError as exc:
            log.error("Telegram channel disabled: %s", exc)
    if ch.email.enabled:
        try:
            notifiers.append(EmailNotifier(os.environ.get(ch.email.smtp_host_env, ""), ch.email.smtp_port,
                                           os.environ.get(ch.email.username_env, ""), os.environ.get(ch.email.password_env, ""),
                                           ch.email.sender, ch.email.recipients, ch.email.use_tls,
                                           min_severity=ch.email.min_severity, max_per_minute=ch.email.max_per_minute))
        except ValueError as exc:
            log.error("Email channel disabled: %s", exc)
    if ch.webhook.enabled:
        webhook = WebhookNotifier(ch.webhook.url, os.environ.get(ch.webhook.token_env) or None, ch.webhook.timeout_s,
                                  ch.webhook.heartbeat_url, min_severity=ch.webhook.min_severity,
                                  max_per_minute=ch.webhook.max_per_minute)
        notifiers.append(webhook)
    return notifiers, overlay, webhook
