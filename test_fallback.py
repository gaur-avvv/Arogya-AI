#!/usr/bin/env python3
"""
Test script for the Arogya AI offline fallback mechanism.
Tests normal flow, direct fallback, and simulated LLM failure.
"""

import sys
import unittest
from tests.test_fallback_mechanism import TestFallbackMechanism

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("=" * 60)
    print("AROGYA AI - OFFLINE FALLBACK MECHANISM TEST SUITE")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFallbackMechanism)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
