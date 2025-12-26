import json
import time
from pathlib import Path
from typing import Dict, Optional


DEFAULT_RESULTS_DIR = Path("results")
ARB_EVENT_TYPES = {"arb_opportunity", "arb_trade", "arb_partial_fill", "arb_skip"}


def create_run_dir(base: Path = DEFAULT_RESULTS_DIR, prefix: str = "run") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class MetricsLogger:
    """Lightweight JSONL metrics writer for training/backtest/live runs."""

    def __init__(self, run_dir: Path, filename: str = "metrics.jsonl"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self._fh = self.path.open("a", encoding="utf-8")

    def log_event(self, event_type: str, payload: Optional[Dict] = None, **kwargs) -> None:
        payload = payload or {}
        record = {"timestamp": time.time(), "type": event_type}
        record.update(payload)
        record.update(kwargs)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
