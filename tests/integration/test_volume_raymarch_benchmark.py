import unittest

from scripts.benchmarks.gpu.benchmark_volume_raymarch import parse_viewer_log


class VolumeRaymarchBenchmarkTest(unittest.TestCase):
    def test_parses_final_gpu_and_session_summaries(self):
        result = parse_viewer_log(
            "[vulkan][perf] Session: 301 frames in 10.0s (30.1 avg FPS)\n"
            "[vulkan][perf] GPU render: samples=301 total=240.80ms avg=0.80ms\n"
        )
        self.assertEqual(result["frames"], 301)
        self.assertEqual(result["gpu_samples"], 301)
        self.assertAlmostEqual(result["gpu_average_ms"], 0.8)

    def test_rejects_vacuous_log(self):
        with self.assertRaises(ValueError):
            parse_viewer_log("GPU timing unavailable")


if __name__ == "__main__":
    unittest.main()
