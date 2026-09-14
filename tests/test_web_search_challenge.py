"""WEB-SEARCH-PROVIDER-CHALLENGE-1.

A DuckDuckGo anti-bot challenge page (HTTP 202 + a `duckduckgo.com/anomaly.js`
form) must NOT be reported as a legitimate successful empty search. It is
detected provider-specifically (by the anomaly marker, never by "0 results"
alone) and surfaced as an informative provider-challenge error.

Fallback decision: no clean keyless fallback exists on this host (DDG Lite serves
the identical challenge; Mojeek gates with a captcha), so this is detection-only
(a provider TECHNICAL_STOP on fallback). These tests therefore assert that a
challenge contacts no second provider and never loops.

Model-free: the network is stubbed with a scripted ``urlopen``; ``time.sleep`` is
patched so no real delay or internet is required.
"""

from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit.runtime import web
from orbit.runtime.web import _is_ddg_challenge, search_web


_RESULT_HTML = """
<div class="result">
  <a class="result__a" href="https://example.org">Example</a>
  <a class="result__snippet">A short snippet.</a>
</div>
"""

# Minimal faithful fragment of the real HTTP 202 challenge page (see mission diag).
_CHALLENGE_HTML = (
    '<!DOCTYPE html><html lang="en"><head><title>DuckDuckGo</title></head><body>'
    '<form id="anomaly-modal__form" '
    'action="//duckduckgo.com/anomaly.js?sv=html&cc=botnet&ti=1&gk=x&q=Dante">'
    '</form></body></html>'
)

# A legitimate zero-result page: no results, and crucially NO anomaly marker.
_EMPTY_HTML = '<html><body><div class="no-results">No results found.</div></body></html>'


class _FakeResponse:
    def __init__(self, html: str, status: int = 200) -> None:
        self._html = html.encode("utf-8")
        self._status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self._status

    def read(self, size: int) -> bytes:
        del size
        return self._html


def _challenge_response() -> _FakeResponse:
    """The real challenge is served with HTTP 202 and the anomaly marker."""
    return _FakeResponse(_CHALLENGE_HTML, status=202)


class WebSearchChallengeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sleeps: list[float] = []
        sleep_patch = mock.patch.object(web.time, "sleep", side_effect=self.sleeps.append)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def _run(self, behaviors: list[object]):
        with mock.patch.object(web, "urlopen") as urlopen:
            urlopen.side_effect = behaviors
            self.urlopen = urlopen
            result = search_web("Dante Alighieri")
        return result

    def test_t1_normal_results_unchanged(self) -> None:
        result = self._run([_FakeResponse(_RESULT_HTML)])
        self.assertIn("web_search_results: true", result)
        self.assertIn("url: https://example.org", result)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_t2_legitimate_empty_stays_success_empty(self) -> None:
        result = self._run([_FakeResponse(_EMPTY_HTML)])
        self.assertEqual(result, "web_search_results: true\nresults: none")
        self.assertFalse(result.startswith("error:"))

    def test_t3_http202_challenge_is_not_success_empty(self) -> None:
        result = self._run([_challenge_response()])
        self.assertNotEqual(result, "web_search_results: true\nresults: none")
        self.assertTrue(result.startswith("error:"))
        self.assertIn("anti-bot challenge", result)

    def test_t4_detection_requires_202_status_and_provider_marker(self) -> None:
        # P3 signature: HTTP 202 AND the provider-anchored anomaly marker. Neither the
        # marker without 202, nor a 202 without the marker, is a challenge; and "0
        # results" is never the trigger.
        self.assertTrue(_is_ddg_challenge(202, _CHALLENGE_HTML))
        self.assertFalse(_is_ddg_challenge(200, _CHALLENGE_HTML))  # marker but 200 -> not a challenge
        self.assertFalse(_is_ddg_challenge(202, _EMPTY_HTML))      # 202 but no marker -> not a challenge
        self.assertFalse(_is_ddg_challenge(200, _EMPTY_HTML))
        self.assertFalse(_is_ddg_challenge(200, _RESULT_HTML))

    def test_t4b_normal_200_results_mentioning_anomaly_are_not_a_challenge(self) -> None:
        # False-positive guard: a legitimate 200 results page whose result URL/snippet
        # references "anomaly.js" (e.g. searching for the anomaly.js library) must be
        # returned as real results, NOT swallowed as a provider challenge.
        results_page = """
        <div class="result">
          <a class="result__a" href="https://github.com/example/anomaly.js">anomaly.js</a>
          <a class="result__snippet">A JS anomaly-detection library at duckduckgo.com/anomaly.js mirror.</a>
        </div>
        """
        self.assertFalse(_is_ddg_challenge(200, results_page))
        result = self._run([_FakeResponse(results_page, status=200)])
        self.assertIn("web_search_results: true", result)
        self.assertIn("anomaly.js", result)
        self.assertNotIn("anti-bot challenge", result)

    def test_t5_challenge_contacts_no_second_provider(self) -> None:
        # No fallback provider exists: a challenge is a single-provider outcome; the
        # challenge page is an HTTP success (not an exception), so there is no retry
        # and no second/other endpoint is contacted.
        result = self._run([_challenge_response()])
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertTrue(result.startswith("error:"))
        # Only the fixed DDG endpoint was ever requested.
        (called_request,), _ = self.urlopen.call_args
        self.assertIn("html.duckduckgo.com", called_request.full_url)

    def test_t8_permanent_and_security_errors_are_not_challenge(self) -> None:
        # A permanent 4xx surfaces the retry/permanent error, never the challenge path.
        perm = HTTPError("https://html.duckduckgo.com/html/", 404, "Not Found", hdrs=None, fp=None)
        result = self._run([perm])
        self.assertIn("web search failed after 1 attempt(s)", result)
        self.assertNotIn("anti-bot challenge", result)

    def test_t9_analysis_network_deny_opens_zero_sockets(self) -> None:
        with mock.patch.object(web, "network_retrieval_denied", return_value=True):
            with mock.patch.object(web, "urlopen") as urlopen:
                result = search_web("Dante Alighieri")
                self.assertEqual(urlopen.call_count, 0)
        self.assertIn(web.ANALYSIS_NETWORK_DENIED_REASON, result)
        self.assertNotIn("anti-bot challenge", result)

    def test_t10_transient_retry_semantics_unchanged(self) -> None:
        # ECONNRESET still retries then succeeds; challenge detection did not alter this.
        result = self._run([URLError(ConnectionResetError(104, "reset")), _FakeResponse(_RESULT_HTML)])
        self.assertEqual(self.urlopen.call_count, 2)
        self.assertIn("web_search_results: true", result)
        self.assertEqual(len(self.sleeps), 1)

    def test_t10b_transient_then_challenge_does_not_loop(self) -> None:
        # A retry that recovers into a challenge page is classified as a challenge, not
        # retried further: total attempts stay within the bounded retry policy.
        result = self._run([URLError(ConnectionResetError(104, "reset")), _challenge_response()])
        self.assertEqual(self.urlopen.call_count, 2)  # 1 transient + 1 that returns the challenge
        self.assertTrue(result.startswith("error:"))
        self.assertIn("anti-bot challenge", result)

    def test_t11_no_provider_loop_on_challenge(self) -> None:
        # Even if the provider would keep challenging, detection happens once on the
        # returned body; there is no re-fetch loop.
        result = self._run([_challenge_response()])
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertTrue(result.startswith("error:"))

    def test_t12_provider_identity_observable_in_challenge(self) -> None:
        result = self._run([_challenge_response()])
        self.assertIn("duckduckgo", result.lower())


if __name__ == "__main__":
    unittest.main()
