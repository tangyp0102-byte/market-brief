"""Shared HTTP helper: retries, timeouts, credential redaction, raw archiving.

Every raw response is written to data/raw/ before parsing. Free sources break in
undignified ways (HTML error pages served with a 200, silently truncated CSVs),
and when a number looks wrong weeks later the raw payload is the only way to
tell whether the source lied or the parser did.

SECRETS: API keys travel in query strings for FRED, so any error or log line
containing a URL would otherwise leak the key into your terminal, your CI logs,
and anywhere you paste them for help. Every outbound message goes through
redact() first.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
BACKOFF_SECONDS = 2.0

USER_AGENT = "market-brief/1.0 (personal research tool)"

# Matches key=value pairs in a query string whose name looks like a credential.
_SENSITIVE_QS = re.compile(
    r"((?:api_key|apikey|api-key|token|access_token|secret|password)=)[^&\s)\"']+",
    re.IGNORECASE,
)


class FetchError(RuntimeError):
    """Raised when a source cannot be fetched after retries."""


def redact(text) -> str:
    """Mask credential-looking query parameters in any string.

    Applied to every log line and exception message. Without this, a single
    failed FRED call prints your API key to the console.
    """
    return _SENSITIVE_QS.sub(r"\1<REDACTED>", str(text))


def fetch_text(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    raw_dir: str | Path | None = None,
    label: str = "fetch",
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> str:
    """GET a URL and return the body text, retrying on transient failures.

    `headers` is merged over the default User-Agent, so individual sources can
    override it where a plain default is rejected.
    """
    last_message: str | None = None
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=request_headers,
            )

            if response.status_code >= 400:
                raise FetchError(
                    f"{label}: HTTP {response.status_code} from {redact(response.url)}"
                )

            body = response.text

            if not body.strip():
                raise FetchError(f"{label}: empty body from {redact(response.url)}")

            # A 200 containing an HTML error page is the classic silent failure.
            head = body.lstrip()[:200].lower()
            if head.startswith(("<!doctype html", "<html")):
                raise FetchError(
                    f"{label}: got an HTML page where data was expected "
                    f"({redact(response.url)}). Either the endpoint format changed "
                    "or the server is refusing this client."
                )

            if raw_dir is not None:
                _archive(body, raw_dir, label)
            return body

        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_message = redact(exc)
            if attempt < retries:
                sleep_for = BACKOFF_SECONDS * attempt
                log.warning(
                    "%s: attempt %d/%d failed (%s), retrying in %.1fs",
                    label, attempt, retries, last_message, sleep_for,
                )
                time.sleep(sleep_for)

    raise FetchError(f"{label}: failed after {retries} attempts: {last_message}")


def _archive(body: str, raw_dir: str | Path, label: str) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    path = raw_dir / f"{safe_label}_{digest}.txt"
    if not path.exists():
        path.write_text(body)
    return path
