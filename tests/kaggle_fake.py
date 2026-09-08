"""An in-memory stand-in for api.kaggle.com and the signed storage URLs, for httpx.MockTransport.

Mirrors the wire format verified on 2026-09-05: JSON POSTs per method, default-valued fields
omitted, 302 to storage for downloads, raw PUT for uploads, PENDING then COMPLETE polling.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd

SLUG = "playground-series-s6e9"
SERVICE = "/v1/competitions.CompetitionApiService/"
STORAGE = "https://storage.googleapis.com"


def make_bundle(
    n_train: int = 60,
    n_test: int = 25,
    seed: int = 0,
    target: str = "purchased",
    labels: tuple[object, object] = (0, 1),
) -> bytes:
    rng = np.random.default_rng(seed)
    train = pd.DataFrame(
        {
            "id": np.arange(n_train),
            "age": rng.integers(18, 80, n_train),
            "income": rng.normal(50_000, 15_000, n_train).round(2),
            "region": rng.choice(["north", "south", "east"], n_train),
            target: np.asarray(labels, dtype=object)[rng.integers(0, 2, n_train)],
        }
    )
    test = train.drop(columns=[target]).iloc[:n_test].reset_index(drop=True)
    test["id"] = np.arange(n_train, n_train + n_test)
    sample = pd.DataFrame({"id": test["id"], target: 0.5})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("train.csv", train.to_csv(index=False))
        zf.writestr("test.csv", test.to_csv(index=False))
        zf.writestr("sample_submission.csv", sample.to_csv(index=False))
    return buf.getvalue()


class FakeKaggle:
    def __init__(
        self,
        *,
        slug: str = SLUG,
        metric: str | None = "Roc Auc Score",
        bundle: bytes | None = None,
        user_has_entered: bool = True,
        submissions_disabled: bool = False,
        max_daily: int = 10,
        submissions: list[dict] | None = None,
        upload_503s: int = 0,
        pending_polls: int = 1,
        public_score: float = 0.5,
        fail_submission: str | None = None,
        page_size: int | None = None,
        code_only: bool = False,
    ) -> None:
        self.slug = slug
        self.metric = metric
        self.bundle = make_bundle() if bundle is None else bundle
        self.user_has_entered = user_has_entered
        self.submissions_disabled = submissions_disabled
        self.max_daily = max_daily
        self.submissions: list[dict] = list(submissions or [])
        self.upload_503s = upload_503s
        self.pending_polls = pending_polls
        self.public_score = public_score
        self.fail_submission = fail_submission
        self.page_size = page_size
        self.code_only = code_only
        self.calls: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.tokens: list[str] = []
        self._next_ref = 1000

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def competition_item(self) -> dict:
        item = {
            "id": 125219,
            "ref": f"https://www.kaggle.com/competitions/{self.slug}",
            "title": "Predicting Electric Vehicle Purchases",
            "deadline": "2026-09-30T23:59:00Z",
            "maxDailySubmissions": self.max_daily,
            "userHasEntered": self.user_has_entered,
            "submissionsDisabled": self.submissions_disabled,
            "isKernelsSubmissionsOnly": self.code_only,
        }
        if self.metric:
            item["evaluationMetric"] = self.metric
        return item

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        self.calls.append(f"{request.method} {host}{path}")
        if host == "storage.googleapis.com":
            return self._storage(request)
        if host != "api.kaggle.com" or not path.startswith(SERVICE):
            return httpx.Response(404, text=f"unexpected {request.url}")
        if "authorization" not in request.headers or "ka_sessionid" in request.headers.get(
            "cookie", ""
        ):
            # A returned session cookie makes the gateway ignore the bearer token (seen live).
            return httpx.Response(
                401,
                json={
                    "error": {
                        "code": 401,
                        "message": "Unauthenticated",
                        "status": "UNAUTHENTICATED",
                    }
                },
            )
        method = path[len(SERVICE) :]
        body = json.loads(request.content or b"{}")
        fn = getattr(self, f"_{method}", None)
        if fn is None:
            return httpx.Response(404, json={"code": 404, "message": f"no method {method}"})
        resp = fn(body)
        resp.headers["set-cookie"] = "ka_sessionid=abc123; Path=/; HttpOnly"
        return resp

    # -- service methods --------------------------------------------------------------------

    def _ListCompetitions(self, body: dict) -> httpx.Response:
        decoy = {"ref": f"https://www.kaggle.com/competitions/{self.slug}9", "title": "Decoy"}
        return httpx.Response(200, json={"competitions": [decoy, self.competition_item()]})

    def _ListDataFiles(self, body: dict) -> httpx.Response:
        assert body["competitionName"] == self.slug
        files = [
            {"name": n, "totalBytes": 10}
            for n in ("sample_submission.csv", "test.csv", "train.csv")
        ]
        return httpx.Response(200, json={"files": files})

    def _DownloadDataFiles(self, body: dict) -> httpx.Response:
        assert body["competitionName"] == self.slug
        return httpx.Response(302, headers={"Location": f"{STORAGE}/bundle/archive.zip?sig=1"})

    def _ListSubmissions(self, body: dict) -> httpx.Response:
        assert body["competitionName"] == self.slug
        for sub in self.submissions:
            if sub.get("status") == "PENDING":
                sub["_polls"] = sub.get("_polls", self.pending_polls) - 1
                if sub["_polls"] < 0:
                    if self.fail_submission:
                        sub["status"] = "ERROR"
                        sub["errorDescription"] = self.fail_submission
                    else:
                        sub["status"] = "COMPLETE"
                        sub["publicScore"] = str(self.public_score)
        visible = [{k: v for k, v in s.items() if not k.startswith("_")} for s in self.submissions]
        if not visible:
            return httpx.Response(200, json={})
        if self.page_size:
            start = int(body.get("pageToken") or 0)
            page = visible[start : start + self.page_size]
            payload = {"submissions": page}
            if start + self.page_size < len(visible):
                payload["nextPageToken"] = str(start + self.page_size)
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"submissions": visible})

    def _StartSubmissionUpload(self, body: dict) -> httpx.Response:
        assert body["competitionName"] == self.slug
        assert (
            body["fileName"] and body["contentLength"] > 0 and body["lastModifiedEpochSeconds"] > 0
        )
        token = f"tok-{len(self.tokens) + 1}"
        self.tokens.append(token)
        return httpx.Response(200, json={"token": token, "createUrl": f"{STORAGE}/upload/{token}"})

    def _CreateSubmission(self, body: dict) -> httpx.Response:
        assert body["competitionName"] == self.slug
        token = body["blobFileTokens"]
        assert token in self.tokens, "finalise before start"
        assert token in self.uploads, "finalise before upload"
        self._next_ref += 1
        self.submissions.append(
            {
                "ref": self._next_ref,
                "date": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "description": body.get("submissionDescription", ""),
                "fileName": "submission.csv",
                "status": "PENDING",
                "totalBytes": len(self.uploads[token]),
            }
        )
        return httpx.Response(
            200, json={"message": "Successfully submitted", "ref": self._next_ref}
        )

    # -- storage ---------------------------------------------------------------------------

    def _storage(self, request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers, "credentials leaked to storage host"
        assert "cookie" not in request.headers, "cookies leaked to storage host"
        if request.method == "GET" and request.url.path == "/bundle/archive.zip":
            if "content-type" in request.headers:
                return httpx.Response(403, text="<Error><Code>SignatureDoesNotMatch</Code></Error>")
            return httpx.Response(
                200, content=self.bundle, headers={"Content-Length": str(len(self.bundle))}
            )
        if request.method == "PUT" and request.url.path.startswith("/upload/"):
            if self.upload_503s > 0:
                self.upload_503s -= 1
                return httpx.Response(503, text="try again")
            self.uploads[request.url.path.rsplit("/", 1)[-1]] = request.content
            return httpx.Response(200)
        return httpx.Response(404, text=f"unexpected storage {request.url}")
