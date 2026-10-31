"""8-Hour periodic background synchronization daemon and scheduler."""

import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional
from ..database.models import log_sync_event
from ..engines.arxiv_miner import ArxivMiner
from ..engines.grant_miner import GrantMiner
from ..engines.inspire_hep import InspireHepIngestor


class SyncDaemon:
    """Manages periodic updates across ArXiv, job boards, and grant feeds."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config["database"]["path"]
        self.interval_seconds = int(config.get("sync", {}).get("interval_hours", 8)) * 3600
        self.arxiv_miner = ArxivMiner(config)
        self.job_ingestor = InspireHepIngestor(config)
        self.grant_miner = GrantMiner(config)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run_sync_cycle(self) -> Dict[str, int]:
        """
        Executes a complete update cycle across all data streams,
        logging timing and record counts to SQLite.
        """
        results = {"arxiv": 0, "positions": 0, "grants": 0}
        cycle_start = datetime.now()

        # 1. Sync ArXiv Preprints
        t0 = time.time()
        start_str = datetime.now().isoformat()
        try:
            results["arxiv"] = self.arxiv_miner.sync()
            log_sync_event(
                self.db_path,
                source="arxiv",
                records_fetched=results["arxiv"],
                status="success",
                message=f"Fetched {results['arxiv']} new ArXiv preprints",
                started_at=start_str,
                finished_at=datetime.now().isoformat(),
                duration_sec=round(time.time() - t0, 2),
            )
        except Exception as e:
            log_sync_event(
                self.db_path,
                source="arxiv",
                records_fetched=0,
                status="warning",
                message=f"ArXiv sync warning: {e}",
                started_at=start_str,
                finished_at=datetime.now().isoformat(),
                duration_sec=round(time.time() - t0, 2),
            )

        # 2. Sync Job Positions
        t0 = time.time()
        start_str = datetime.now().isoformat()
        try:
            results["positions"] = self.job_ingestor.sync()
            log_sync_event(
                self.db_path,
                source="inspirehep_jobs",
                records_fetched=results["positions"],
                status="success",
                message=f"Fetched {results['positions']} new job openings",
                started_at=start_str,
                finished_at=datetime.now().isoformat(),
                duration_sec=round(time.time() - t0, 2),
            )
        except Exception as e:
            log_sync_event(
                self.db_path,
                source="inspirehep_jobs",
                records_fetched=0,
                status="warning",
                message=f"Jobs sync warning: {e}",
                started_at=start_str,
                finished_at=datetime.now().isoformat(),
                duration_sec=round(time.time() - t0, 2),
            )

        # 3. Sync Research Grants
        t0 = time.time()
        start_str = datetime.now().isoformat()
        try:
            results["grants"] = self.grant_miner.sync()
            log_sync_event(
                self.db_path,
                source="erc_grants",
                records_fetched=results["grants"],
                status="success",
                message=f"Synced {results['grants']} new research grants",
                started_at=start_str,
                finished_at=datetime.now().isoformat(),
                duration_sec=round(time.time() - t0, 2),
            )
        except Exception as e:
            log_sync_event(
                self.db_path,
                source="erc_grants",
                records_fetched=0,
                status="warning",
                message=f"Grants sync warning: {e}",
                started_at=start_str,
                finished_at=datetime.now().isoformat(),
                duration_sec=round(time.time() - t0, 2),
            )

        total_duration = round((datetime.now() - cycle_start).total_seconds(), 2)
        print(f"[SyncDaemon] Cycle completed in {total_duration}s: {results}")
        return results

    def _daemon_loop(self):
        """Internal loop sleeping for the interval duration."""
        print(f"[SyncDaemon] Started background runner (interval: {self.interval_seconds // 3600} hours).")
        while not self._stop_event.is_set():
            try:
                self.run_sync_cycle()
            except Exception as e:
                print(f"[SyncDaemon] Cycle exception: {e}")
            # Sleep in small increments to allow rapid clean exit on stop_event
            for _ in range(self.interval_seconds // 5):
                if self._stop_event.is_set():
                    break
                time.sleep(5)

    def start_background(self) -> None:
        """Starts the daemon loop in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            print("[SyncDaemon] Already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._daemon_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signals background thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
            print("[SyncDaemon] Stopped.")
