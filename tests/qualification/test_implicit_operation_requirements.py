import unittest

from scripts.tests.catalog.qualify_catalog_bounded_runs import require_operation_counts


class ImplicitOperationRequirementsTests(unittest.TestCase):
    def setUp(self):
        self.stencil = "diffusion/implicit_diffusion_stencil.comp"
        self.vector = "diffusion/implicit_diffusion_vector.comp"
        self.required = {"kernelCounts": {"dynamics/slow_apply.comp": 10},
                         "implicitSolver": {"solves": 25, "maximumIterations": 4096}}

    def counts(self, stencil=100, vector=150):
        return {"dynamics/slow_apply.comp": 10, self.stencil: stencil, self.vector: vector}

    def test_valid_iterations_may_differ_without_changing_ownership(self):
        require_operation_counts(self.counts(), self.required)
        require_operation_counts(self.counts(125, 200), self.required)

    def test_missing_residual_verification_rejects(self):
        with self.assertRaises(RuntimeError):
            require_operation_counts(self.counts(75, 150), self.required)

    def test_missing_or_extra_stage_rejects(self):
        for counts in (self.counts(100, 151), {**self.counts(), "unexpected.comp": 1}):
            with self.assertRaises(RuntimeError):
                require_operation_counts(counts, self.required)

    def test_exhausted_iteration_budget_rejects(self):
        iterations = 25 * 4096 + 1
        with self.assertRaises(RuntimeError):
            require_operation_counts(self.counts(25 + iterations + 25, 50 + 2 * iterations), self.required)

    def test_fixed_operations_remain_exact(self):
        changed = self.counts()
        changed["dynamics/slow_apply.comp"] = 9
        with self.assertRaises(RuntimeError):
            require_operation_counts(changed, self.required)
        with self.assertRaises(RuntimeError):
            require_operation_counts({"a.comp": 2}, {"kernelCounts": {"a.comp": 1}})


if __name__ == "__main__":
    unittest.main()
