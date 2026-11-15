from contextlib import redirect_stderr
import io
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import check_core


class CoreCheckDriverTests(unittest.TestCase):
    def run_cases(self, cases):
        with mock.patch.object(sys, "path", list(sys.path)), \
             mock.patch.object(unittest.defaultTestLoader, "loadTestsFromNames",
                               return_value=unittest.TestSuite(cases)), \
             redirect_stderr(io.StringIO()):
            return check_core.run_suite()

    def test_successful_nonempty_suite_passes(self):
        self.assertEqual(self.run_cases([unittest.FunctionTestCase(lambda: None)]), 0)

    def test_empty_suite_and_skipped_coverage_fail(self):
        self.assertEqual(self.run_cases([]), 1)
        skipped = unittest.FunctionTestCase(unittest.skip("missing artifact")(lambda: None))
        self.assertEqual(self.run_cases([skipped]), 1)

    def test_failed_tests_cannot_be_masked_by_the_wrapper(self):
        def fail():
            raise AssertionError("fixture failure")
        self.assertEqual(self.run_cases([unittest.FunctionTestCase(fail)]), 1)

    def test_parent_returns_the_real_fresh_child_exit_code(self):
        run = mock.Mock(return_value=SimpleNamespace(returncode=7))
        with mock.patch.dict(sys.modules, {"qualify_fresh": SimpleNamespace(run_fresh=run)}):
            self.assertEqual(check_core.main([]), 7)
        self.assertEqual(run.call_args.args[0][-1], "--run-suite")

    def test_unknown_arguments_do_not_start_another_interpreter(self):
        run = mock.Mock()
        with mock.patch.dict(sys.modules, {"qualify_fresh": SimpleNamespace(run_fresh=run)}), \
             redirect_stderr(io.StringIO()):
            self.assertEqual(check_core.main(["--unknown"]), 2)
        run.assert_not_called()
