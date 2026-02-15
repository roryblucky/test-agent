"""Gunicorn configuration for K8s deployment.

Launch with::

    gunicorn app.main:app -c gunicorn.conf.py

For 4-core / 8GB K8s pods:
- Single worker (async handles concurrency; multiple workers duplicate memory)
- uvloop for ~20-30% async scheduling speedup
- Worker recycling to prevent memory leaks over long runs
- Keep-alive matched to K8s ingress (typically 60s)
"""

import os

# --- Server ---
bind = os.environ.get("BIND", "0.0.0.0:8000")

# --- Workers ---
# Single async worker per pod — scale horizontally via K8s HPA.
# Multiple workers duplicate the 8GB memory footprint unnecessarily.
workers = int(os.environ.get("WORKERS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# --- Timeouts ---
timeout = 300  # Kill worker if stuck > 5 min
graceful_timeout = 30  # Grace period on SIGTERM
keepalive = 65  # Match K8s ingress (usually 60s)

# --- Worker recycling (prevent slow memory leaks) ---
max_requests = 5000
max_requests_jitter = 500

# --- Logging ---
accesslog = "-"  # stdout
loglevel = os.environ.get("LOG_LEVEL", "info")

# --- uvloop (high-performance event loop) ---
# Uvicorn's UvicornWorker auto-detects and uses uvloop if installed.
# No explicit configuration needed here — just ensure uvloop is in deps.
