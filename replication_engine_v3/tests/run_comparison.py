#!/usr/bin/env python3
"""Run replication tests with a specified model and provider."""
import sys
import os

model_name = sys.argv[1] if len(sys.argv) > 1 else "glm-5:cloud"
provider = sys.argv[2] if len(sys.argv) > 2 else "ollama_cloud"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_replication

test_replication.MODEL_NAME = model_name
test_replication.PROVIDER = provider
test_replication.main()
