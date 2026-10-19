from __future__ import annotations

import math
import unittest

from scripts.qualification.checkpointing.qualify_vulkan_restart import require_identity


class SlowMomentumClosureOracleTest(unittest.TestCase):
    def test_roundoff_uses_work_operands_not_near_zero_residual(self) -> None:
        committed = 72779.50390625
        requested = -98443.73688326037
        storage_rounding = 171223.24078951037
        residual = 7.275957614183426e-12
        require_identity(
            0.0, residual, "static vertical-only row 5", "slow-momentum closure",
            reference_values=(committed, requested, storage_rounding),
        )

    def test_more_than_64_operand_ulps_still_rejects(self) -> None:
        committed = 72779.50390625
        requested = -98443.73688326037
        storage_rounding = 171223.24078951037
        scale = max(abs(committed), abs(requested), abs(storage_rounding))
        breach = 65 * math.ulp(scale)
        with self.assertRaisesRegex(RuntimeError, "slow-momentum closure identity failed"):
            require_identity(
                0.0, breach, "mutation", "slow-momentum closure",
                reference_values=(committed, requested, storage_rounding),
            )


if __name__ == "__main__":
    unittest.main()
