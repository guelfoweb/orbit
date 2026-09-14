"""WEB-SEARCH-RELIABILITY-1.

Bounded retry for CHAT web search on genuinely transient provider/network
failures (ECONNRESET/timeout/temporary DNS/429/selected 5xx). Permanent errors,
programming errors, and security/policy denials must NOT retry, the retry bound
must hold, and the final error must stay informative.

Model-free: the network is stubbed with a scripted ``urlopen`` and ``time.sleep``
is patched so no real delay or internet is required.
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
from orbit.runtime.web import SEARCH_MAX_ATTEMPTS, execute_fetch_url, search_web


_RESULT_HTML = """
<div class="result">
  <a class="result__a" href="https://example.org">Example</a>
  <a class="result__snippet">A short snippet.</a>
</div>
"""


class _FakeResponse:
    def __init__(self, html: str) -> None:
        self._html = html.encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        del size
        return self._html


def _reset_urlerror() -> URLError:
    """The exact observed failure: URLError wrapping ECONNRESET."""
    return URLError(ConnectionResetError(104, "Connection reset by peer"))


class WebSearchRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        # No real backoff delay in the unit gate; record calls to prove bounded backoff.
        self.sleeps: list[float] = []
        sleep_patch = mock.patch.object(web.time, "sleep", side_effect=self.sleeps.append)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def _run(self, behaviors: list[object]):
        with mock.patch.object(web, "urlopen") as urlopen:
            urlopen.side_effect = behaviors
            result = search_web("Dante Alighieri")
        return result, urlopen.call_count

    def test_t1_success_path_unchanged(self) -> None:
        result, calls = self._run([_FakeResponse(_RESULT_HTML)])
        self.assertEqual(calls, 1)
        self.assertIn("web_search_results: true", result)
        self.assertIn("url: https://example.org", result)
        self.assertEqual(self.sleeps, [])  # no backoff on first-attempt success

    def test_t2_econnreset_triggers_bounded_retry(self) -> None:
        result, calls = self._run([_reset_urlerror(), _FakeResponse(_RESULT_HTML)])
        self.assertEqual(calls, 2)
        self.assertIn("web_search_results: true", result)
        self.assertEqual(len(self.sleeps), 1)  # one backoff between the two attempts

    def test_t2_bare_connection_reset_also_retries(self) -> None:
        result, calls = self._run([ConnectionResetError(104, "reset"), _FakeResponse(_RESULT_HTML)])
        self.assertEqual(calls, 2)
        self.assertIn("web_search_results: true", result)

    def test_t3_timeout_triggers_bounded_retry(self) -> None:
        for exc in (socket.timeout("timed out"), TimeoutError("timed out"), URLError(socket.timeout("t"))):
            with self.subTest(exc=type(exc).__name__):
                self.sleeps.clear()
                result, calls = self._run([exc, _FakeResponse(_RESULT_HTML)])
                self.assertEqual(calls, 2)
                self.assertIn("web_search_results: true", result)

    def test_t3_temporary_dns_failure_retries(self) -> None:
        eai_again = getattr(socket, "EAI_AGAIN", -3)
        transient_dns = URLError(socket.gaierror(eai_again, "Temporary failure in name resolution"))
        result, calls = self._run([transient_dns, _FakeResponse(_RESULT_HTML)])
        self.assertEqual(calls, 2)
        self.assertIn("web_search_results: true", result)

    def test_t4_retryable_http_statuses_retry(self) -> None:
        for code in (429, 500, 502, 503, 504):
            with self.subTest(code=code):
                self.sleeps.clear()
                err = HTTPError("https://html.duckduckgo.com/html/", code, "transient", hdrs=None, fp=None)
                result, calls = self._run([err, _FakeResponse(_RESULT_HTML)])
                self.assertEqual(calls, 2)
                self.assertIn("web_search_results: true", result)

    def test_t5_permanent_4xx_does_not_retry(self) -> None:
        for code in (400, 403, 404, 451):
            with self.subTest(code=code):
                self.sleeps.clear()
                err = HTTPError("https://html.duckduckgo.com/html/", code, "permanent", hdrs=None, fp=None)
                result, calls = self._run([err, _FakeResponse(_RESULT_HTML)])
                self.assertEqual(calls, 1)  # no retry
                self.assertIn("error: web search failed after 1 attempt(s)", result)
                self.assertEqual(self.sleeps, [])  # permanent failure not delayed

    def test_t5_permanent_dns_and_refused_do_not_retry(self) -> None:
        eai_noname = getattr(socket, "EAI_NONAME", -2)
        for exc in (
            URLError(socket.gaierror(eai_noname, "Name or service not known")),
            URLError(ConnectionRefusedError(111, "Connection refused")),
        ):
            with self.subTest(exc=repr(exc)):
                self.sleeps.clear()
                result, calls = self._run([exc, _FakeResponse(_RESULT_HTML)])
                self.assertEqual(calls, 1)
                self.assertIn("error: web search failed after 1 attempt(s)", result)

    def test_t6_programming_errors_do_not_retry(self) -> None:
        # A non-OSError (e.g. a parser/programming bug) is not caught by the retry
        # loop; it propagates on the first attempt and is never retried.
        with mock.patch.object(web, "urlopen") as urlopen:
            urlopen.side_effect = [ValueError("boom"), _FakeResponse(_RESULT_HTML)]
            with self.assertRaises(ValueError):
                search_web("Dante Alighieri")
            self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(self.sleeps, [])

    def test_t7_retry_limit_is_enforced(self) -> None:
        # Transient on every attempt -> exactly SEARCH_MAX_ATTEMPTS attempts, no more.
        result, calls = self._run([_reset_urlerror()] * (SEARCH_MAX_ATTEMPTS + 5))
        self.assertEqual(calls, SEARCH_MAX_ATTEMPTS)
        self.assertEqual(len(self.sleeps), SEARCH_MAX_ATTEMPTS - 1)
        self.assertIn(f"after {SEARCH_MAX_ATTEMPTS} attempt(s)", result)

    def test_t8_final_error_is_informative_after_exhaustion(self) -> None:
        result, _ = self._run([_reset_urlerror()] * SEARCH_MAX_ATTEMPTS)
        self.assertTrue(result.startswith("error: web search failed after"))
        self.assertIn("Connection reset by peer", result)  # original error preserved

    def test_t9_successful_retry_returns_normal_results(self) -> None:
        result, calls = self._run([_reset_urlerror(), _reset_urlerror(), _FakeResponse(_RESULT_HTML)])
        self.assertEqual(calls, 3)
        self.assertIn("web_search_results: true", result)
        self.assertIn("title: Example", result)
        self.assertIn("snippet: A short snippet.", result)

    def test_t10_analysis_network_deny_is_immediate_and_never_retries(self) -> None:
        with mock.patch.object(web, "network_retrieval_denied", return_value=True):
            with mock.patch.object(web, "urlopen") as urlopen:
                result = search_web("Dante Alighieri")
                self.assertEqual(urlopen.call_count, 0)  # no socket opened at all
        self.assertIn(web.ANALYSIS_NETWORK_DENIED_REASON, result)
        self.assertEqual(self.sleeps, [])

    def test_t11_security_denial_short_circuits_before_retry_loop(self) -> None:
        # The policy gate is checked before any attempt: retry can never re-drive a
        # denied request. fetch_url (the user-URL / SSRF surface) likewise denies
        # immediately and is untouched by this change.
        with mock.patch.object(web, "network_retrieval_denied", return_value=True):
            with mock.patch.object(web, "urlopen") as urlopen:
                fetch_result = execute_fetch_url({"url": "http://169.254.169.254/latest/meta-data/"})
                self.assertEqual(urlopen.call_count, 0)
        self.assertIn("status: policy_denied", fetch_result)

    def test_t12_retry_is_http_level_single_tool_result(self) -> None:
        # A transient-then-success run makes multiple HTTP attempts but yields exactly
        # one tool result string: retry is internal to search_web and introduces no
        # extra tool/model invocation.
        result, calls = self._run([_reset_urlerror(), _FakeResponse(_RESULT_HTML)])
        self.assertEqual(calls, 2)
        self.assertEqual(result.count("web_search_results: true"), 1)


if __name__ == "__main__":
    unittest.main()
