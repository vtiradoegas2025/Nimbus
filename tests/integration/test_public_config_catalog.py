#!/usr/bin/env python3
"""Static contract for the public resolution-first configuration catalog."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "scripts/dev/configs/generate_resolution_catalog.py"
SPEC = importlib.util.spec_from_file_location("public_config_generator", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class PublicConfigCatalogTests(unittest.TestCase):
    def test_required_resolution_and_variant_matrix(self) -> None:
        expected_directories = {
            f"{resolution}meterResolution" for resolution in generator.RESOLUTIONS_M
        }
        actual_directories = {
            path.name for path in generator.CONFIG_ROOT.iterdir() if path.is_dir()
        }
        self.assertTrue(expected_directories.issubset(actual_directories))
        for directory_name in expected_directories:
            resolution_directory = generator.CONFIG_ROOT / directory_name
            microphysics_directories = {
                path.name for path in resolution_directory.iterdir() if path.is_dir()
            }
            self.assertTrue(set(generator.SCHEMES).issubset(microphysics_directories))
            for scheme in generator.SCHEMES:
                files = {
                    path.name for path in (resolution_directory / scheme).iterdir()
                    if path.is_file()
                }
                required_files = {
                    generator.profile_filename(profile)
                    for profile in generator.PROFILES
                }
                self.assertTrue(required_files.issubset(files))

    def test_checked_in_catalog_matches_generator(self) -> None:
        self.assertEqual(generator.generate(check=True), 0)

    def test_every_config_is_standalone_and_resolution_consistent(self) -> None:
        required_sections = (
            "schema: nimbus.v1",
            "grid:",
            "time:",
            "domain_motion:",
            "microphysics:",
            "initial_conditions:",
            "output:",
            "extensions:",
        )
        for resolution in generator.RESOLUTIONS_M:
            for variant in generator.SCHEMES:
                for profile in generator.PROFILES:
                    path = (
                        generator.CONFIG_ROOT
                        / f"{resolution}meterResolution"
                        / variant
                        / generator.profile_filename(profile)
                    )
                    text = path.read_text(encoding="utf-8")
                    for section in required_sections:
                        self.assertIn(section, text, path)
                    self.assertRegex(text, rf"(?m)^  dx: {resolution}$")
                    self.assertRegex(text, rf"(?m)^  dy: {resolution}$")
                    self.assertRegex(text, r"(?m)^  keep_last_frames: [1-9][0-9]*$")
                    self.assertRegex(text, r"(?m)^  minimum_free_space_mib: [1-9][0-9]*$")
                    self.assertNotIn("acknowledge_maturity_limits:", text, path)
                    ceiling = re.search(
                        r"(?m)^  maximum_retained_history_files: ([1-9][0-9]*)$",
                        text,
                    )
                    self.assertIsNotNone(ceiling, path)
                    self.assertGreaterEqual(int(ceiling.group(1)), 51, path)
                    self.assertLessEqual(int(ceiling.group(1)), 4096, path)
                    expected_scheme = "none" if variant == "dry" else variant
                    match = re.search(
                        r"(?m)^  scheme: (\S+)$",
                        text[text.index("microphysics:"):],
                    )
                    self.assertIsNotNone(match, path)
                    self.assertEqual(match.group(1), expected_scheme, path)

    def test_profile_physics_cross_product(self) -> None:
        expected = {
            "baseline": ("smagorinsky", "2d", "slab", "slab", "none"),
            "radiation": ("smagorinsky", "2d", "slab", "slab", "simple_grey"),
            "ysu": ("smagorinsky", "2d", "slab", "ysu", "none"),
            "ysu_radiation": ("smagorinsky", "2d", "slab", "ysu", "simple_grey"),
            "les": ("smagorinsky", "3d", "none", "none", "none"),
            "les_radiation": ("smagorinsky", "3d", "none", "none", "simple_grey"),
            "tke": ("tke", "3d", "none", "none", "none"),
            "tke_radiation": ("tke", "3d", "none", "none", "simple_grey"),
        }
        for profile, schemes in expected.items():
            text = generator.render_config(1000, "kessler", profile)
            blocks = {
                name: text.split(f"{name}:\n", 1)[1].split("\n\n", 1)[0]
                for name in ("turbulence", "surface", "boundary_layer", "radiation")
            }
            turbulence, mode, surface, pbl, radiation = schemes
            self.assertIn(f"  scheme: {turbulence}", blocks["turbulence"])
            self.assertIn(f"  mode: {mode}", blocks["turbulence"])
            self.assertIn(f"  scheme: {surface}", blocks["surface"])
            self.assertIn(f"  scheme: {pbl}", blocks["boundary_layer"])
            self.assertIn(f"  scheme: {radiation}", blocks["radiation"])


if __name__ == "__main__":
    unittest.main()
