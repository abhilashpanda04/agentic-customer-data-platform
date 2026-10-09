"""Shared test configuration.

Sets a fixed approval-token secret and short TTL so token tests are
deterministic and do not emit RuntimeWarnings about the ephemeral secret.
"""
from __future__ import annotations

import os

os.environ.setdefault("CDP_APPROVAL_SECRET", "test-only-secret-not-for-production")
os.environ.setdefault("CDP_APPROVAL_TTL_MINUTES", "30")