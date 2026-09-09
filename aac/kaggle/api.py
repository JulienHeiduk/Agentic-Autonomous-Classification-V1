"""Raw HTTP client for the Kaggle API (README section 6).

Wire format verified 2026-09-05 against the official client (kaggle 2.2.4, kagglesdk 0.1.37)
and the live service. Every call is ``POST {base_url}/{service}/{Method}`` with a camelCase
JSON body and either a ``KGAT_`` bearer token or legacy basic auth. Response fields at their
default value are omitted, so an empty submission list comes back as ``{}``.

Data download: ``DownloadDataFiles`` answers 302 to a signed storage URL, which is fetched
with a bare GET (no credentials, no Content-Type: the V2 signature covers that header, so a
JSON content type carried over from the POST yields SignatureDoesNotMatch). Submission:
``StartSubmissionUpload`` returns an upload URL and a blob token, the file is PUT raw to that
URL, then ``CreateSubmission`` finalises with the token.

Cookies are never sent back: api.kaggle.com sets ``ka_sessionid`` on every response, and a
request carrying it is treated as an anonymous session (HTTP 401) even with a valid bearer
token. The jar is cleared before every request.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

KAGGLE_API = "https://api.kaggle.com/v1"
COMPETITIONS_SERVICE = "competitions.CompetitionApiService"
_MAX_PAGES = 50


class KaggleError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class KaggleAuthError(KaggleError):
    """No usable credentials, or Kaggle rejected them."""


@dataclass(frozen=True)
class CompetitionInfo:
    slug: str
    title: str
    metric_name: str | None
    deadline: str | None
    max_daily_submissions: int | None
    user_has_entered: bool
    submissions_disabled: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class DataFile:
    name: str
    total_bytes: int
    creation_date: str | None = None


@dataclass(frozen=True)
class Submission:
    ref: int
    date: str | None
    description: str
    file_name: str
    status: str  # PENDING, COMPLETE, ERROR
    public_score: float | None
    private_score: float | None
    error_description: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def complete(self) -> bool:
        return self.status == "COMPLETE"

    @property
    def failed(self) -> bool:
        return self.status == "ERROR" or bool(self.error_description)

    @property
    def settled(self) -> bool:
        return self.complete or self.failed

    @classmethod
    def from_api(cls, item: dict[str, Any]) -> Submission:
        return cls(
            ref=int(item.get("ref") or 0),
            date=item.get("date") or None,
            description=str(item.get("description") or ""),
            file_name=str(item.get("fileName") or ""),
            status=_normalise_status(item.get("status")),
            public_score=_score(item.get("publicScore")),
            private_score=_score(item.get("privateScore")),
            error_description=str(item.get("errorDescription") or ""),
            raw=item,
        )


def _score(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_status(value: Any) -> str:
    # The enum serialises as "COMPLETE" today; tolerate a "SUBMISSION_STATUS_" prefix.
    text = str(value or "PENDING").upper()
    return text.rsplit("_", 1)[-1]


class KaggleClient:
    def __init__(
        self,
        *,
        access_token: str | None = None,
        username: str | None = None,
        key: str | None = None,
        base_url: str = KAGGLE_API,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if access_token:
            self.auth_mode = "bearer"
            self._headers = {"Authorization": f"Bearer {access_token}"}
            self._auth: httpx.Auth | None = None
        elif username and key:
            self.auth_mode = "basic"
            self._headers = {}
            self._auth = httpx.BasicAuth(username, key)
        else:
            raise KaggleAuthError(
                "no Kaggle credentials: set KAGGLE_ACCESS_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY)"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sleep = sleep
        self._http = client or httpx.Client(timeout=timeout)
        self._owned = client is None

    @classmethod
    def from_env(cls, **kwargs: Any) -> KaggleClient:
        return cls(
            access_token=os.environ.get("KAGGLE_ACCESS_TOKEN"),
            username=os.environ.get("KAGGLE_USERNAME"),
            key=os.environ.get("KAGGLE_KEY"),
            **kwargs,
        )

    def close(self) -> None:
        if self._owned:
            self._http.close()

    def __enter__(self) -> KaggleClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"KaggleClient(auth={self.auth_mode}, base_url={self.base_url!r})"

    # -- transport ------------------------------------------------------------------------

    def _url(self, method: str, service: str = COMPETITIONS_SERVICE) -> str:
        return f"{self.base_url}/{service}/{method}"

    def _forget_cookies(self) -> None:
        # See module docstring: a returned ka_sessionid cookie overrides bearer auth.
        self._http.cookies.clear()

    @staticmethod
    def _check(resp: httpx.Response, what: str) -> None:
        status, message = resp.status_code, resp.text[:300]
        if "application/json" in resp.headers.get("content-type", ""):
            try:
                envelope = resp.json()
            except ValueError:
                envelope = None
            if isinstance(envelope, dict) and isinstance(envelope.get("error"), dict):
                envelope = envelope["error"]
            code = envelope.get("code") if isinstance(envelope, dict) else None
            if isinstance(code, int) and code >= 400:
                status = code
                message = str(envelope.get("message") or message)
        if status in (401, 403):
            raise KaggleAuthError(f"{what}: HTTP {status}: {message}", status=status, body=message)
        if status >= 400:
            raise KaggleError(f"{what}: HTTP {status}: {message}", status=status, body=message)

    def call(
        self, method: str, body: dict[str, Any], *, service: str = COMPETITIONS_SERVICE
    ) -> dict[str, Any]:
        url = self._url(method, service)
        self._forget_cookies()
        try:
            resp = self._http.post(
                url, json=body, headers=self._headers, auth=self._auth, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise KaggleError(f"{method}: {exc.__class__.__name__}: {exc}") from exc
        self._check(resp, method)
        try:
            data = resp.json()
        except ValueError as exc:
            raise KaggleError(f"{method}: non-JSON response: {resp.text[:200]}") from exc
        if not isinstance(data, dict):
            raise KaggleError(f"{method}: expected a JSON object, got {type(data).__name__}")
        return data

    def _paginate(self, method: str, body: dict[str, Any], key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        token: str | None = None
        for _ in range(_MAX_PAGES):
            page = dict(body)
            if token:
                page["pageToken"] = token
            data = self.call(method, page)
            items.extend(x for x in (data.get(key) or []) if isinstance(x, dict))
            token = data.get("nextPageToken") or None
            if not token:
                return items
        raise KaggleError(f"{method}: more than {_MAX_PAGES} pages")

    # -- competitions ---------------------------------------------------------------------

    def competition(self, slug: str) -> CompetitionInfo:
        """Metadata for one competition, including the evaluation metric."""
        data = self.call("ListCompetitions", {"search": slug})
        for item in data.get("competitions") or []:
            ref = str(item.get("ref") or item.get("url") or "").rstrip("/")
            if ref.rsplit("/", 1)[-1] == slug:
                return CompetitionInfo(
                    slug=slug,
                    title=str(item.get("title") or slug),
                    metric_name=item.get("evaluationMetric") or None,
                    deadline=item.get("deadline") or None,
                    max_daily_submissions=item.get("maxDailySubmissions"),
                    user_has_entered=bool(item.get("userHasEntered", False)),
                    submissions_disabled=bool(item.get("submissionsDisabled", False)),
                    raw=item,
                )
        raise KaggleError(f"competition {slug!r} not found in search results")

    def list_files(self, slug: str) -> list[DataFile]:
        items = self._paginate("ListDataFiles", {"competitionName": slug}, "files")
        return [
            DataFile(str(i.get("name")), int(i.get("totalBytes") or 0), i.get("creationDate"))
            for i in items
        ]

    def download_all(self, slug: str, dest: Path) -> Path:
        """Stream the competition bundle to ``dest``. Atomic: a partial file never survives."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        self._forget_cookies()
        try:
            first = self._http.post(
                self._url("DownloadDataFiles"),
                json={"competitionName": slug},
                headers=self._headers,
                auth=self._auth,
                follow_redirects=False,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise KaggleError(f"DownloadDataFiles: {exc.__class__.__name__}: {exc}") from exc
        self._check(first, "DownloadDataFiles")
        if first.is_redirect:
            location = first.headers.get("location")
            if not location:
                raise KaggleError("DownloadDataFiles: redirect without a Location header")
            request = self._http.build_request(
                "GET", location, timeout=httpx.Timeout(self.timeout, read=600.0)
            )
            request.headers.pop("content-type", None)
            request.headers.pop("authorization", None)
            request.headers.pop("cookie", None)
        else:
            request = None  # the bundle came back inline

        received = 0
        expected = 0
        try:
            if request is None:
                expected = int(first.headers.get("content-length") or 0)
                tmp.write_bytes(first.content)
                received = len(first.content)
            else:
                resp = self._http.send(request, stream=True, follow_redirects=True)
                try:
                    if resp.status_code >= 400:
                        resp.read()
                        raise KaggleError(
                            f"download: HTTP {resp.status_code}: {resp.text[:200]}",
                            status=resp.status_code,
                            body=resp.text[:500],
                        )
                    expected = int(resp.headers.get("content-length") or 0)
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_bytes(1 << 20):
                            fh.write(chunk)
                            received += len(chunk)
                finally:
                    resp.close()
        except httpx.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            raise KaggleError(f"download: {exc.__class__.__name__}: {exc}") from exc
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if expected and received != expected:
            tmp.unlink(missing_ok=True)
            raise KaggleError(f"download: got {received} bytes, expected {expected}")
        os.replace(tmp, dest)
        log.info("downloaded %s (%d bytes) to %s", slug, received, dest)
        return dest

    # -- submissions ----------------------------------------------------------------------

    def list_submissions(self, slug: str) -> list[Submission]:
        items = self._paginate("ListSubmissions", {"competitionName": slug}, "submissions")
        return [Submission.from_api(i) for i in items]

    def start_upload(self, slug: str, file: Path) -> tuple[str, str]:
        """Step 1 of the handshake: returns (blob token, upload URL)."""
        stat = file.stat()
        data = self.call(
            "StartSubmissionUpload",
            {
                "competitionName": slug,
                "fileName": file.name,
                "contentLength": stat.st_size,
                "lastModifiedEpochSeconds": int(stat.st_mtime),
            },
        )
        token, url = data.get("token"), data.get("createUrl")
        if not token or not url:
            raise KaggleError(f"StartSubmissionUpload: no token/createUrl in {data}")
        return str(token), str(url)

    def upload_blob(self, create_url: str, payload: bytes, *, attempts: int = 5) -> None:
        """Step 2: raw PUT to the signed URL, no credentials. 503 is retried with backoff."""
        delay = 1.0
        for attempt in range(1, attempts + 1):
            self._forget_cookies()
            try:
                resp = self._http.put(
                    create_url,
                    content=payload,
                    headers={"Content-Length": str(len(payload))},
                    timeout=httpx.Timeout(self.timeout, write=600.0),
                )
            except httpx.HTTPError as exc:
                if attempt == attempts:
                    raise KaggleError(f"upload: {exc.__class__.__name__}: {exc}") from exc
                self._sleep(delay)
                delay *= 2
                continue
            if resp.status_code in (200, 201):
                return
            if resp.status_code == 503 and attempt < attempts:
                self._sleep(delay)
                delay *= 2
                continue
            raise KaggleError(
                f"upload: HTTP {resp.status_code}: {resp.text[:200]}",
                status=resp.status_code,
                body=resp.text[:500],
            )

    def create_submission(self, slug: str, token: str, description: str) -> int:
        """Step 3: finalise. Returns the submission ref."""
        data = self.call(
            "CreateSubmission",
            {
                "competitionName": slug,
                "blobFileTokens": token,
                "submissionDescription": description,
            },
        )
        ref = int(data.get("ref") or 0)
        if not ref:
            raise KaggleError(f"CreateSubmission: no ref returned: {data.get('message') or data}")
        log.info("submission %d created: %s", ref, data.get("message", ""))
        return ref

    def submit(self, slug: str, file: Path, description: str) -> int:
        file = Path(file)
        token, url = self.start_upload(slug, file)
        self.upload_blob(url, file.read_bytes())
        return self.create_submission(slug, token, description)

    def get_submission(self, slug: str, ref: int) -> Submission | None:
        for sub in self.list_submissions(slug):
            if sub.ref == ref:
                return sub
        return None

    def wait_for_score(
        self,
        slug: str,
        ref: int,
        *,
        timeout: float = 600.0,
        interval: float = 15.0,
        raise_on_timeout: bool = True,
    ) -> Submission:
        """Poll the submission list until ``ref`` is COMPLETE or ERROR. When the timeout
        passes first: raise, or with ``raise_on_timeout`` False return the pending row (Kaggle
        keeps scoring; the score is fetched later)."""
        deadline = time.monotonic() + timeout
        last: Submission | None = None
        while True:
            last = self.get_submission(slug, ref)
            if last is not None and last.settled:
                return last
            if time.monotonic() >= deadline:
                state = last.status if last else "not listed"
                if raise_on_timeout:
                    raise KaggleError(f"submission {ref} still {state} after {timeout:.0f}s")
                return last or Submission(ref, None, "", "", "PENDING", None, None, "")
            self._sleep(interval)
