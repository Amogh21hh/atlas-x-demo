"""Material-event notifications. Posts to a webhook (Slack/Discord/ntfy-compatible JSON) if configured."""
from __future__ import annotations

import logging

import requests

from .config import settings

log = logging.getLogger("atlas.notify")


def notify(level: str, kind: str, message: str) -> None:
    log.warning("[%s] %s: %s", level, kind, message)
    if not settings.notify_webhook:
        return
    try:
        requests.post(settings.notify_webhook, json={"text": f"ATLAS X [{level}] {kind}: {message}"}, timeout=10)
    except Exception as e:  # notification failure must never break trading logic
        log.error("notify failed: %s", e)
