from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import math
import re
from typing import Any
from urllib import error, request
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, unquote

from .common import ToolError, coerce_int


MAX_FETCH_TEXT_CHARS = 12_000
MAX_FETCH_LINKS = 25
MAX_SEARCH_RESULTS = 10
DEFAULT_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
BROWSER_LIKE_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
LINK_RE = re.compile(r"""<a\s+[^>]*href=["']([^"'#]+)["']""", re.IGNORECASE)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
HIGHLIGHT_PATTERNS = (
    re.compile(r".{0,60}temperatura minima.{0,120}?massima.{0,80}?(?:°|&deg;|&#176;)\s*C.{0,40}", re.IGNORECASE),
    re.compile(r".{0,60}temperatura massima.{0,120}?minima.{0,80}?(?:°|&deg;|&#176;)\s*C.{0,40}", re.IGNORECASE),
    re.compile(r".{0,40}(?:sereno|nuvoloso|pioggia|temporale|sole).{0,120}?(?:°|&deg;|&#176;)\s*C.{0,40}", re.IGNORECASE),
)
RESULT_LINK_RE = re.compile(
    r"""<a[^>]*class=["'][^"']*result__a[^"']*["'][^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)
RESULT_BLOCK_RE = re.compile(
    r"""<div[^>]*class=["'][^"']*result[^"']*["'][^>]*>(.*?)</div>\s*</div>""",
    re.IGNORECASE | re.DOTALL,
)
RESULT_SNIPPET_RE = re.compile(
    r"""<(?:a|div)[^>]*class=["'][^"']*result__snippet[^"']*["'][^>]*>(.*?)</(?:a|div)>""",
    re.IGNORECASE | re.DOTALL,
)
RESULT_URL_RE = re.compile(
    r"""<a[^>]*class=["'][^"']*result__url[^"']*["'][^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)


def web_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": (
                    "Search the web for an open-ended query and return bounded results with title, URL, and snippet. "
                    "Use this for generic online research when you do not already have a concrete URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_RESULTS},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 120},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": (
                    "Fetch one explicit web page URL and return text, title, final URL, and a bounded link list. "
                    "Use this only when you already have a concrete http/https URL. "
                    "Use start_char to continue a long page in chunks. "
                    "Do not use it as a general web search tool and do not guess search-engine or encyclopedia URLs from a name alone."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "start_char": {"type": "integer", "minimum": 0},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": MAX_FETCH_TEXT_CHARS},
                        "max_links": {"type": "integer", "minimum": 0, "maximum": MAX_FETCH_LINKS},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 120},
                    },
                    "required": ["url"],
                },
            },
        }
    ]


@dataclass
class WebTools:
    search_endpoint: str = DEFAULT_SEARCH_ENDPOINT

    def search_web(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query is required")
        max_results = min(MAX_SEARCH_RESULTS, max(1, coerce_int(arguments.get("max_results"), 5)))
        timeout = min(120, max(1, coerce_int(arguments.get("timeout"), 20)))
        html = self._request_text(
            url=self.search_endpoint,
            timeout=timeout,
            form_data={"q": query.strip()},
        )
        results = self._extract_search_results(html, max_results=max_results)
        return {
            "ok": True,
            "query": query.strip(),
            "provider": "duckduckgo-html",
            "results": results,
        }

    def fetch_url(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_url = arguments.get("url")
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise ToolError("url is required")
        parsed = urlparse(raw_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ToolError("url must be a valid http or https URL")
        start_char = max(0, coerce_int(arguments.get("start_char"), 0))
        max_chars = min(MAX_FETCH_TEXT_CHARS, max(200, coerce_int(arguments.get("max_chars"), 8000)))
        max_links = min(MAX_FETCH_LINKS, max(0, coerce_int(arguments.get("max_links"), 10)))
        timeout = min(120, max(1, coerce_int(arguments.get("timeout"), 20)))
        final_url, status_code, content_type, text = self._fetch_page(raw_url, timeout=timeout)
        full_text = self._extract_text(text)
        total_chars = len(full_text)
        if start_char > total_chars:
            start_char = total_chars
        end_char = min(total_chars, start_char + max_chars)
        chunk_text = full_text[start_char:end_char]
        highlights = self._extract_highlights(chunk_text)
        truncated = end_char < total_chars
        chunk_count = max(1, math.ceil(total_chars / max_chars)) if total_chars else 1
        chunk_index = 1 if not total_chars else (start_char // max_chars) + 1
        return {
            "ok": True,
            "url": raw_url,
            "final_url": final_url,
            "status_code": status_code,
            "content_type": content_type,
            "title": self._extract_title(text),
            "text": chunk_text,
            "highlights": highlights,
            "links": self._extract_links(text, final_url, max_links),
            "start_char": start_char,
            "end_char": end_char,
            "total_chars": total_chars,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "next_start_char": end_char if truncated else None,
            "has_more": truncated,
            "truncated": truncated,
        }

    def _fetch_page(
        self,
        raw_url: str,
        *,
        timeout: int,
        form_data: dict[str, str] | None = None,
    ) -> tuple[str, int | None, str | None, str]:
        data = None
        headers = {
            "User-Agent": BROWSER_LIKE_USER_AGENT,
            "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        if form_data:
            data = urlencode(form_data).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = request.Request(
            raw_url,
            data=data,
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                final_url = response.geturl()
                status_code = getattr(response, "status", None)
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
                body = response.read()
        except error.HTTPError as exc:
            raise ToolError(f"http error: {exc.code}") from exc
        except error.URLError as exc:
            raise ToolError(f"connection failed: {exc.reason}") from exc
        try:
            text = body.decode(charset, errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")
        return final_url, status_code, content_type, text

    def _request_text(self, url: str, *, timeout: int, form_data: dict[str, str] | None = None) -> str:
        _, _, _, text = self._fetch_page(url, timeout=timeout, form_data=form_data)
        return text

    @classmethod
    def _extract_search_results(cls, html: str, *, max_results: int) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for block in RESULT_BLOCK_RE.findall(html):
            link_match = RESULT_LINK_RE.search(block)
            if not link_match:
                continue
            raw_href = unescape(link_match.group(1)).strip()
            url = cls._normalize_result_url(raw_href)
            if not url or url in seen:
                continue
            seen.add(url)
            title = cls._clean_fragment(link_match.group(2))
            snippet_match = RESULT_SNIPPET_RE.search(block)
            snippet = cls._clean_fragment(snippet_match.group(1)) if snippet_match else ""
            display_match = RESULT_URL_RE.search(block)
            display_url = cls._clean_fragment(display_match.group(1)) if display_match else ""
            result = {
                "title": title or url,
                "url": url,
                "snippet": snippet,
            }
            if display_url:
                result["display_url"] = display_url
            results.append(result)
            if len(results) >= max_results:
                break
        if results:
            return results
        for href, title_html in RESULT_LINK_RE.findall(html):
            url = cls._normalize_result_url(unescape(href).strip())
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                {
                    "title": cls._clean_fragment(title_html) or url,
                    "url": url,
                    "snippet": "",
                }
            )
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _normalize_result_url(raw_url: str) -> str | None:
        if not raw_url:
            return None
        parsed = urlparse(raw_url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/y.js"):
                return None
            if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
                target = parse_qs(parsed.query).get("uddg")
                if target:
                    decoded = unquote(target[0]).strip()
                    nested = urlparse(decoded)
                    if nested.scheme in {"http", "https"} and nested.netloc:
                        return decoded
            return raw_url
        return None

    @staticmethod
    def _extract_title(text: str) -> str | None:
        match = TITLE_RE.search(text)
        if not match:
            return None
        title = WHITESPACE_RE.sub(" ", unescape(match.group(1))).strip()
        return title or None

    @staticmethod
    def _extract_links(text: str, base_url: str, max_links: int) -> list[str]:
        if max_links <= 0:
            return []
        links: list[str] = []
        seen: set[str] = set()
        for href in LINK_RE.findall(text):
            resolved = urljoin(base_url, unescape(href).strip())
            parsed = urlparse(resolved)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or resolved in seen:
                continue
            seen.add(resolved)
            links.append(resolved)
            if len(links) >= max_links:
                break
        return links

    @staticmethod
    def _extract_text(text: str) -> str:
        without_scripts = SCRIPT_STYLE_RE.sub(" ", text)
        without_tags = TAG_RE.sub(" ", without_scripts)
        return WHITESPACE_RE.sub(" ", unescape(without_tags)).strip()

    @staticmethod
    def _extract_highlights(text: str, *, limit: int = 5) -> list[str]:
        if not text:
            return []
        highlights: list[str] = []
        seen: set[str] = set()
        for pattern in HIGHLIGHT_PATTERNS:
            for match in pattern.findall(text):
                cleaned = WHITESPACE_RE.sub(" ", unescape(match)).strip(" -")
                if len(cleaned) < 20 or cleaned in seen:
                    continue
                seen.add(cleaned)
                highlights.append(cleaned)
                if len(highlights) >= limit:
                    return highlights
        return highlights

    @classmethod
    def _clean_fragment(cls, text: str) -> str:
        return cls._extract_text(text)
