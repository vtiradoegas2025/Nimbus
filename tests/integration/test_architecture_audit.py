import unittest

from scripts.benchmarks.audit_architecture_paths import telemetry


BASE = '[PERF SUMMARY] measured_steps=16, avg_step_ms=10\n[DYNAMICS PERF] dynamics_total_s=0.16\n'


class ArchitectureTelemetryTests(unittest.TestCase):
    def test_absent_gpu_timing_is_unknown(self):
        row = telemetry(BASE, 'metal')
        self.assertFalse(row['gpu_timing_complete'])
        self.assertIsNone(row['gpu'])
        self.assertIsNone(row['peak_rss_bytes'])

    def test_partial_or_legacy_vulkan_timestamps_are_not_complete(self):
        for samples in ('', ' timestamped_command_buffers=1'):
            text = BASE + '[GPU STAGE TELEMETRY] acoustic command_buffers=2' + samples + '\n'
            text += '[GPU SUBMIT TIMING] acoustic 0.5 0.9 2 0.25 0.45\n'
            self.assertFalse(telemetry(text, 'vulkan')['gpu_timing_complete'])

    def test_complete_vulkan_timing_preserves_transfers(self):
        text = BASE + '[GPU STAGE TELEMETRY] acoustic command_buffers=2 timestamped_command_buffers=2 device_upload_bytes=1024\n'
        text += '[GPU SUBMIT TIMING] acoustic 0.5 0.9 2 0.25 0.45\n'
        row = telemetry(text, 'vulkan')
        self.assertTrue(row['gpu_timing_complete'])
        self.assertEqual(row['gpu_stages']['acoustic']['device_upload_bytes'], 1024)
        self.assertEqual(row['gpu_stages']['acoustic']['fence_wait_ms'], 0.9)

    def test_metal_durations_and_coverage_are_distinct(self):
        text = BASE + '[METAL TELEMETRY] scope=backend_generation command_buffers=2 timestamped_command_buffers=2 gpu_total_ms=0.5 submit_wait_wall_ms=0.9\n'
        row = telemetry(text, 'metal')
        self.assertTrue(row['gpu_timing_complete'])
        self.assertEqual(row['metal_generations'][0]['submit_wait_wall_ms'], 0.9)
        with self.assertRaises(ValueError):
            telemetry(text.replace('gpu_total_ms=0.5', 'gpu_total_ms=-1'), 'metal')


if __name__ == '__main__':
    unittest.main()
