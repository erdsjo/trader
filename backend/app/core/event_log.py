from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EngineEvent:
    timestamp: str
    level: str
    message: str


class EventLog:
    def __init__(self, max_events: int = 200):
        self._events: deque[EngineEvent] = deque(maxlen=max_events)

    def _append(self, level: str, message: str) -> None:
        self._events.append(
            EngineEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                level=level,
                message=message,
            )
        )

    def info(self, message: str) -> None:
        self._append("info", message)

    def warning(self, message: str) -> None:
        self._append("warning", message)

    def error(self, message: str) -> None:
        self._append("error", message)

    def get_events(self, limit: int = 50) -> list[dict]:
        events = list(self._events)[-limit:]
        events.reverse()
        return [
            {"timestamp": e.timestamp, "level": e.level, "message": e.message}
            for e in events
        ]
