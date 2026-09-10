"""Network retrieval is denied during autonomous static analysis.

These tests prove NON-OCCURRENCE with an independent witness: `urlopen` is
monkeypatched to a recorder that raises if it is ever called, so a passing test
is proof that no socket was opened. They also prove the gate is context-scoped
(ordinary CHAT fetch is unaffected) and that a denial is reported as a policy
refusal, never as a network failure or a success.

No real network access is used or needed.
"""

from __future__ import annotations

import unittest

import orbit.runtime.web as web
from orbit.runtime.analysis_network_policy import (
    ANALYSIS_NETWORK_DENIED_REASON,
    analysis_network_denied,
    network_retrieval_denied,
)
from orbit.runtime.web import (
    execute_fetch_url,
    fetch_url_result_status,
    search_web,
)


class _UrlopenWitness:
    """Records every urlopen call and refuses to perform one. Installed in place
    of web.urlopen; any call is a test failure -- the proof of non-occurrence."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("urlopen was called: an outbound request occurred")


class NetworkNonOccurrenceTests(unittest.TestCase):
    def setUp(self):
        self.witness = _UrlopenWitness()
        self._real_urlopen = web.urlopen
        web.urlopen = self.witness

    def tearDown(self):
        web.urlopen = self._real_urlopen

    # A-T1 / A-T2: fetch refused, zero outbound request.
    def test_fetch_refused_no_socket(self):
        with analysis_network_denied():
            result = execute_fetch_url({"url": "https://example.test/payload.exe"})
        self.assertEqual(self.witness.calls, [])  # A-T2 witness: nothing sent
        self.assertEqual(fetch_url_result_status(result), "policy_denied")  # A-T1

    # A-T3: a redirect target cannot bypass the gate -- the request never starts,
    # so urllib's redirect handling is never reached.
    def test_redirect_cannot_bypass(self):
        with analysis_network_denied():
            execute_fetch_url({"url": "https://example.test/redirector"})
        self.assertEqual(self.witness.calls, [])

    # A-T4: URL variants (scheme case, host case, trailing dot, IP form) all denied.
    def test_url_variants_all_denied(self):
        variants = [
            "HTTPS://Example.TEST/a",
            "http://example.test./a",
            "https://93.184.216.34/a",
            "https://example.test:8443/a",
            "https://example.test/a?x=1#frag",
        ]
        with analysis_network_denied():
            for url in variants:
                with self.subTest(url=url):
                    result = execute_fetch_url({"url": url})
                    self.assertEqual(fetch_url_result_status(result), "policy_denied")
        self.assertEqual(self.witness.calls, [])

    # A-T5: localhost / private targets are denied under the same contract (no
    # documented exception exists for static analysis).
    def test_localhost_and_private_denied(self):
        for url in (
            "http://127.0.0.1/x",
            "http://localhost:9000/x",
            "http://10.0.0.5/x",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url):
                with analysis_network_denied():
                    result = execute_fetch_url({"url": url})
                self.assertEqual(fetch_url_result_status(result), "policy_denied")
        self.assertEqual(self.witness.calls, [])

    # A-T6: a blocked request produces no retrieved-body evidence -- no text.
    def test_no_retrieved_payload_evidence(self):
        with analysis_network_denied():
            result = execute_fetch_url({"url": "https://example.test/p"})
        self.assertNotIn("text:\n", result)  # no body was produced
        self.assertIn(ANALYSIS_NETWORK_DENIED_REASON, result)

    # A-T3-variant: search_web is an outbound request and is denied too.
    def test_search_denied(self):
        with analysis_network_denied():
            result = search_web("anything at all")
        self.assertEqual(self.witness.calls, [])
        self.assertIn(ANALYSIS_NETWORK_DENIED_REASON, result)

    # A-T9: CHAT (no analysis context) is baseline-equivalent -- the gate is
    # inert, so the request proceeds to urlopen (the witness proves it is
    # reached; in production it performs the real fetch).
    def test_chat_fetch_reaches_network(self):
        self.assertFalse(network_retrieval_denied())
        with self.assertRaises(AssertionError):  # witness fires on the real attempt
            execute_fetch_url({"url": "https://example.test/ok"})
        self.assertEqual(len(self.witness.calls), 1)

    # A-T11: direct internal dispatch cannot bypass -- calling the executor
    # directly under the context is still denied (the gate is at the network
    # entry, not the tool grammar).
    def test_direct_dispatch_denied(self):
        with analysis_network_denied():
            result = execute_fetch_url({"url": "https://example.test/direct"})
        self.assertEqual(fetch_url_result_status(result), "policy_denied")
        self.assertEqual(self.witness.calls, [])

    # A-T8: repeated attempts each refuse deterministically, none open a socket.
    def test_repeated_attempts_bounded(self):
        with analysis_network_denied():
            for _ in range(5):
                result = execute_fetch_url({"url": "https://example.test/x"})
                self.assertEqual(fetch_url_result_status(result), "policy_denied")
        self.assertEqual(self.witness.calls, [])

    # A-T10: the context restores on exit -- denial does not leak past the run.
    def test_context_restores(self):
        self.assertFalse(network_retrieval_denied())
        with analysis_network_denied():
            self.assertTrue(network_retrieval_denied())
        self.assertFalse(network_retrieval_denied())

    # Nesting is re-entrant and restores correctly.
    def test_nested_context(self):
        with analysis_network_denied():
            with analysis_network_denied():
                self.assertTrue(network_retrieval_denied())
            self.assertTrue(network_retrieval_denied())
        self.assertFalse(network_retrieval_denied())


class DenialSemanticsTests(unittest.TestCase):
    """A denial is a policy refusal, not a backend/network failure, and not a
    success. (No urlopen patch: the denial returns before any network anyway.)"""

    def test_status_is_policy_denied_not_network_error(self):
        with analysis_network_denied():
            result = execute_fetch_url({"url": "https://example.test/x"})
        self.assertEqual(fetch_url_result_status(result), "policy_denied")
        self.assertNotIn("network_error", result)
        self.assertNotIn("timeout", result)

    def test_denied_is_not_success(self):
        with analysis_network_denied():
            result = execute_fetch_url({"url": "https://example.test/x"})
        self.assertNotEqual(fetch_url_result_status(result), "ok")
        self.assertFalse(web.fetch_url_result_has_text(result))


class RunAutonomousEntersDenialTests(unittest.TestCase):
    """run_autonomous holds the denial for its whole lifetime, and releases it
    on exit -- proven by observing the ContextVar from inside the backend's own
    chat calls, which run on the same stack as every step, action and report."""

    def _run(self):
        import tempfile
        from pathlib import Path

        from orbit.runtime.analysis_runtime import AnalysisRuntime, acquire_analysis_source
        from orbit.runtime.evidence import EvidenceStore
        from tests.test_analysis_runtime import ScriptedBackend, prose_response

        observed = []

        class ObservingBackend(ScriptedBackend):
            def chat_stream(self, *a, **k):
                observed.append(network_retrieval_denied())
                return super().chat_stream(*a, **k)

        tmp = tempfile.mkdtemp(prefix="orbit-netpol-")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        original = Path(tmp) / "artifact.txt"
        original.write_text("alpha\nbeta\n", encoding="utf-8")
        source = acquire_analysis_source(original, Path(tmp) / "owned")
        store = EvidenceStore(root=Path(tmp) / "evidence")
        # An empty plan closes the run immediately after PLAN; enough to prove
        # the context is entered before the first model call and released after.
        backend = ObservingBackend(prose_response("done"), plan_questions=[])
        runtime = AnalysisRuntime(backend=backend, source=source, evidence_store=store)
        self.addCleanup(runtime.close)
        self.assertFalse(network_retrieval_denied())  # not entered before the run
        runtime.run_autonomous("Analyse this artifact.")
        return observed

    def test_denial_active_during_every_model_call(self):
        observed = self._run()
        self.assertTrue(observed, "expected at least one model call")
        self.assertTrue(all(observed), "network denial must hold during every call")
        self.assertFalse(network_retrieval_denied())  # released after the run


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
