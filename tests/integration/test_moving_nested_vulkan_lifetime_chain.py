#!/usr/bin/env python3
"""Focused tests for segmented moving-nest Vulkan lifetime inspection."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "qualification" / "nested_domains"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import inspect_moving_nested_vulkan_lifetime_chain as chain


class MovingNestedVulkanLifetimeChainTests(unittest.TestCase):
    def test_segment_bounds_are_exact_and_sortable(self) -> None:
        paths = [
            Path("moving_nested_vulkan_lifetime_v1_20260819_segment_0450_0900"),
            Path("moving_nested_vulkan_lifetime_v1_20260819_segment_0000_0450"),
        ]
        self.assertEqual(chain.segment_bounds(paths[0]), (450.0, 900.0))
        self.assertEqual(
            [chain.segment_bounds(path) for path in sorted(paths, key=chain.segment_bounds)],
            [(0.0, 450.0), (450.0, 900.0)],
        )

    def test_malformed_segment_name_fails_closed(self) -> None:
        with self.assertRaisesRegex(chain.ChainError, "exact bounds"):
            chain.segment_bounds(Path("moving_nested_vulkan_segment_latest"))

    def test_long_segment_enumerates_every_checkpoint_boundary(self) -> None:
        self.assertEqual(
            chain.checkpoint_times(2250.0, 4500.0),
            [2700.0, 3150.0, 3600.0, 4050.0, 4500.0],
        )
        with self.assertRaisesRegex(chain.ChainError, "positive multiple"):
            chain.checkpoint_times(2250.0, 4400.0)

    @unittest.skipUnless(
        chain.DEFAULT_OUTPUT.is_file(), "physical lifetime progress artifact is absent")
    def test_physical_lifetime_chain_revalidates(self) -> None:
        segments = [
            path for path in chain.discover_segments(chain.DEFAULT_GLOB)
            if (path / "moving_nested_vulkan_runtime_report.json").is_file()
        ]
        report = chain.inspect_chain(segments, 4500.0)
        self.assertIn(report["status"], {"in_progress", "complete"})
        self.assertFalse(report["qualification_eligible"])
        self.assertGreaterEqual(report["progress"]["completed_through_s"], 2250.0)
        self.assertGreaterEqual(report["progress"]["tracker_samples"], 75)
        self.assertGreaterEqual(report["progress"]["checkpoint_count"], 5)
        self.assertGreaterEqual(report["progress"]["relocation_count"], 1)
        if report["status"] == "complete":
            self.assertEqual(report["progress"]["completed_through_s"], 4500.0)
            self.assertEqual(report["progress"]["tracker_samples"], 150)
            self.assertEqual(report["progress"]["checkpoint_count"], 10)


if __name__ == "__main__":
    unittest.main()
