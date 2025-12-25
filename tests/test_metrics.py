import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from metrics.logger import MetricsLogger
from metrics.summarize import summarize_run


class MetricsLoggerTests(unittest.TestCase):
    def test_metrics_logger_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            logger = MetricsLogger(run_dir)
            logger.log_event("model_eval", payload={"p_mean": 0.5})
            logger.log_event("decision", payload={"action": "BUY", "size": 1})
            logger.close()

            metrics_path = run_dir / "metrics.jsonl"
            lines = metrics_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["type"], "model_eval")
            self.assertEqual(first["p_mean"], 0.5)

    def test_summarizer_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            logger = MetricsLogger(run_dir)
            logger.log_event("health", payload={"timestamp": 1.0, "equity": 100.0})
            logger.log_event("health", payload={"timestamp": 2.0, "equity": 90.0})
            logger.log_event("filter_skip", payload={"market_id": "m1", "reasons": ["low_volume"]})
            logger.log_event("decision", payload={"action": "BUY", "size": 1})
            logger.log_event("sentiment", payload={"percent_nan": 0.5})
            logger.log_event("trade", payload={"pnl": 1.5})
            logger.close()

            summary = summarize_run(run_dir)
            self.assertEqual(summary["decision_counts"]["BUY"], 1)
            self.assertEqual(summary["filter_skip_counts"]["low_volume"], 1)
            self.assertEqual(summary["total_pnl"], 1.5)
            self.assertTrue(Path(run_dir / "equity.csv").exists())
            dd = pd.read_csv(run_dir / "drawdown.csv")
            self.assertTrue((dd["drawdown"] <= 0).all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
