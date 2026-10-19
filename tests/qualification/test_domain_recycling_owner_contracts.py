from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/tests/catalog/test_domain_recycling_owner_contracts.py"
SPEC = importlib.util.spec_from_file_location("domain_recycling_owners", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
OWNERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNERS)
SOURCE = ROOT / "src/core/infra/domain_recycling_state.cpp"


class DomainRecyclingOwnerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_authoritative_contract_is_complete(self) -> None:
        records = OWNERS.parse_contracts(self.source)
        counts = OWNERS.validate_contracts(records)
        self.assertEqual(len(records), 28)
        self.assertEqual(sum(counts.values()), 28)
        self.assertEqual(counts["InvalidateDeviceResidency"], 1)

    def test_partial_readiness_fails_closed(self) -> None:
        mutated = self.source.replace(
            "Ready::ReadyForTransaction,",
            "Ready::Partial,",
            1,
        )
        records = OWNERS.parse_contracts(mutated)
        with self.assertRaisesRegex(RuntimeError, "not ready for transaction"):
            OWNERS.validate_contracts(records)

    def test_action_drift_fails_closed(self) -> None:
        mutated = self.source.replace(
            '"compute_residency", Action::InvalidateDeviceResidency',
            '"compute_residency", Action::Preserve',
            1,
        )
        records = OWNERS.parse_contracts(mutated)
        with self.assertRaisesRegex(RuntimeError, "action drifted: compute_residency"):
            OWNERS.validate_contracts(records)

    def test_duplicate_identity_fails_closed(self) -> None:
        mutated = self.source.replace(
            '"base_state_profiles", Action::Preserve',
            '"primary_atmosphere", Action::Preserve',
            1,
        )
        records = OWNERS.parse_contracts(mutated)
        with self.assertRaisesRegex(RuntimeError, "identities are duplicated"):
            OWNERS.validate_contracts(records)


if __name__ == "__main__":
    unittest.main()
