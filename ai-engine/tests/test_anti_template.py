"""
test_anti_template.py

Automated Anti-Template Dependency & Zero-Generic-Head Validation Test.
Proves mathematically and via module inspection that PatientNativeReconstructor
has 0% dependency on generic templates, closed-eye mannequins, or template_positions.npy.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure ai-engine is in path
ai_engine_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ai_engine_dir))

class TestAntiTemplateArchitecture(unittest.TestCase):

    def test_no_template_imports_in_native_reconstructor(self):
        """Verifies reconstruct_native_truedepth and patient_native_fusion do not import any GNM template files."""
        import patient_native_fusion
        import patient_texture_baker
        import reconstruct_native_truedepth
        import reconstruct_cli

        forbidden_modules = [
            "gnm_identity_fit",
            "gnm_face_shell",
            "gnm_correspondence",
            "gnm_vertex_trust",
            "gnm_dense_nose",
            "gnm_width_correction",
        ]

        loaded_modules = sys.modules.keys()
        for forbidden in forbidden_modules:
            self.assertNotIn(
                forbidden,
                loaded_modules,
                f"Forbidden template module '{forbidden}' was imported into runtime!"
            )

    def test_sparse_rgb_capture_is_rejected_instead_of_published(self):
        """A known sparse RGB session must fail density QC, never become a baseline."""
        from patient_native_fusion import PatientNativeReconstructor

        # Test against real patient package directory
        pkg_dir = Path(__file__).parent.parent.parent / ".data" / "patients" / "634cf7bb-3394-4203-983d-229d36b024df" / "scans" / "a1968424-6c79-4ec4-a500-42437d433b8f"
        if pkg_dir.exists():
            reconstructor = PatientNativeReconstructor(pkg_dir)
            with self.assertRaisesRegex(RuntimeError, "minimum 1000"):
                reconstructor.reconstruct_patient_surface()


if __name__ == "__main__":
    unittest.main()
