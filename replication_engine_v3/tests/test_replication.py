#!/usr/bin/env python3
"""
Integration-oriented replication test harness.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_agentic_replication_v2 import AgenticReplicationEngineV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "glm-5:cloud"
PROVIDER = "ollama_cloud"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(BASE_DIR, "runs", "integration_tests")
TEST_SET_DIR = os.path.join(BASE_DIR, "test_set")
MAX_ITERATIONS = 100
CONTEXT_WINDOW = 198000

TEST_PAPERS: List[Dict[str, Any]] = [
    {
        "id": "10001",
        "paper_pdf": os.path.join(TEST_SET_DIR, "10001", "paper.pdf"),
        "replication_dir": os.path.join(TEST_SET_DIR, "10001", "replication_package"),
        "description": "Paper 10001 - R-based analysis with CSV data",
    },
    {
        "id": "10011",
        "paper_pdf": os.path.join(TEST_SET_DIR, "10011", "black_women.pdf"),
        "replication_dir": os.path.join(TEST_SET_DIR, "10011", "replication_package"),
        "description": "Paper 10011 - R-based analysis with RData",
    },
    {
        "id": "10090",
        "paper_pdf": os.path.join(TEST_SET_DIR, "10090", "How the Party Commands the Gun.pdf"),
        "replication_dir": os.path.join(TEST_SET_DIR, "10090", "replication_package"),
        "description": "Paper 10090 - R-based analysis with .tab data",
    },
    {
        "id": "10166",
        "paper_pdf": os.path.join(TEST_SET_DIR, "10166", "paper.pdf"),
        "replication_dir": os.path.join(TEST_SET_DIR, "10166"),
        "description": "Paper 10166 - R-based analysis with .rds data",
    },
    {
        "id": "10177",
        "paper_pdf": os.path.join(TEST_SET_DIR, "10177", "paper.pdf"),
        "replication_dir": os.path.join(TEST_SET_DIR, "10177", "replication_package"),
        "description": "Paper 10177 - R-based analysis with CSV data",
    },
]


def run_single_replication(config: Dict[str, Any]) -> Dict[str, Any]:
    paper_id = config["id"]
    logger.info("=" * 70)
    logger.info("TESTING PAPER %s: %s", paper_id, config["description"])
    logger.info("=" * 70)

    start_time = time.time()
    try:
        engine = AgenticReplicationEngineV2(
            model_name=MODEL_NAME,
            provider=PROVIDER,
            runs_root=RUNS_ROOT,
            context_window=CONTEXT_WINDOW,
            max_tokens=8192,
        )
        results = engine.replicate(
            paper_path=config["paper_pdf"],
            replication_package_dir=config["replication_dir"],
            max_iterations=MAX_ITERATIONS,
        )
        elapsed = time.time() - start_time
        return {
            "id": paper_id,
            "success": True,
            "grade": results["grade"],
            "score": results["score"],
            "matches": results["matches"],
            "total_comparisons": results["total_comparisons"],
            "elapsed_seconds": elapsed,
            "comparisons": results["comparisons"],
            "primary_language": results.get("package_inventory", {}).get("primary_language", "Unknown"),
            "summary_path": results["summary_path"],
            "report_tex_path": results["report_tex_path"],
            "error": None,
        }
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.exception("Paper %s FAILED: %s", paper_id, exc)
        return {
            "id": paper_id,
            "success": False,
            "grade": "Failed",
            "score": 0.0,
            "matches": 0,
            "total_comparisons": 0,
            "elapsed_seconds": elapsed,
            "comparisons": [],
            "error": str(exc),
        }


def print_summary(all_results: List[Dict[str, Any]]) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("REPLICATION TEST SUMMARY")
    logger.info("=" * 80)
    logger.info("Model: %s | Provider: %s", MODEL_NAME, PROVIDER)

    for result in all_results:
        logger.info(
            "%s | grade=%s | score=%.1f%% | matches=%d/%d | time=%.1fm",
            result["id"],
            result["grade"],
            result["score"],
            result["matches"],
            result["total_comparisons"],
            result["elapsed_seconds"] / 60,
        )


def main() -> None:
    all_results = [run_single_replication(config) for config in TEST_PAPERS]
    print_summary(all_results)
    os.makedirs(RUNS_ROOT, exist_ok=True)
    summary_path = os.path.join(
        RUNS_ROOT,
        f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": MODEL_NAME,
                "provider": PROVIDER,
                "timestamp": datetime.now().isoformat(),
                "results": all_results,
            },
            handle,
            indent=2,
            default=str,
        )
    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
