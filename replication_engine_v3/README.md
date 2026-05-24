# Replication Engine V3

This directory contains the active implementation of the AI Replication Engine.

Use the repository-level guide for setup, commands, outputs, evidence policy, OCR behavior, and troubleshooting:

[../README.md](../README.md)

Primary entry points:

- `run_agentic_replication_v2.py`: single-paper engine CLI.
- `benchmark_runner.py`: dataset-aware subprocess runner for benchmark batches.
- `core/annotation_engine.py`: SQLite-to-XLSX annotation export helpers.
