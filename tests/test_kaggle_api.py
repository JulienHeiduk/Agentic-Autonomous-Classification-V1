import httpx
import pytest

from aac.kaggle.api import KaggleAuthError, KaggleClient, KaggleError
from tests.kaggle_fake import SLUG, FakeKaggle


def client_for(fake: FakeKaggle, **kw) -> KaggleClient:
    kw.setdefault("access_token", "KGAT_x")
    return KaggleClient(client=httpx.Client(transport=fake.transport), sleep=lambda s: None, **kw)


def test_from_env_prefers_bearer_then_basic_then_fails(monkeypatch):
    for v in ("KAGGLE_ACCESS_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(KaggleAuthError, match="no Kaggle credentials"):
        KaggleClient.from_env()
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    assert KaggleClient.from_env().auth_mode == "basic"
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN", "KGAT_x")
    assert KaggleClient.from_env().auth_mode == "bearer"


def test_competition_metadata():
    fake = FakeKaggle()
    info = client_for(fake).competition(SLUG)
    assert info.slug == SLUG and info.metric_name == "Roc Auc Score"
    assert info.max_daily_submissions == 10 and info.user_has_entered
    assert info.deadline == "2026-09-30T23:59:00Z" and not info.submissions_disabled
    assert fake.calls == [
        "POST api.kaggle.com/v1/competitions.CompetitionApiService/ListCompetitions"
    ]


def test_basic_auth_is_sent():
    fake = FakeKaggle()
    seen = {}
    original = fake.handler

    def spy(request):
        seen["auth"] = request.headers.get("authorization", "")
        return original(request)

    kc = KaggleClient(
        username="u", key="k", client=httpx.Client(transport=httpx.MockTransport(spy))
    )
    kc.competition(SLUG)
    assert seen["auth"].startswith("Basic ")


def test_not_found_and_error_envelopes():
    fake = FakeKaggle(slug="other-comp")
    with pytest.raises(KaggleError, match="not found"):
        client_for(fake).competition(SLUG)

    def unauth(request):
        return httpx.Response(200, json={"code": 401, "message": "Unauthenticated"})

    kc = KaggleClient(access_token="t", client=httpx.Client(transport=httpx.MockTransport(unauth)))
    with pytest.raises(KaggleAuthError, match="Unauthenticated"):
        kc.competition(SLUG)

    def boom(request):
        return httpx.Response(500, text="oops")

    kc = KaggleClient(access_token="t", client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(KaggleError, match="500"):
        kc.competition(SLUG)

    def offline(request):
        raise httpx.ConnectError("no network")

    kc = KaggleClient(access_token="t", client=httpx.Client(transport=httpx.MockTransport(offline)))
    with pytest.raises(KaggleError, match="ConnectError"):
        kc.competition(SLUG)


def test_list_files():
    files = client_for(FakeKaggle()).list_files(SLUG)
    assert [f.name for f in files] == ["sample_submission.csv", "test.csv", "train.csv"]


def test_download_all_follows_redirect_without_credentials(tmp_path):
    fake = FakeKaggle()
    dest = tmp_path / "data" / "archive.zip"
    assert client_for(fake).download_all(SLUG, dest) == dest
    assert dest.read_bytes() == fake.bundle
    assert not dest.with_name("archive.zip.part").exists()
    assert fake.calls[-1] == "GET storage.googleapis.com/bundle/archive.zip"


def test_download_size_mismatch_is_an_error(tmp_path):
    fake = FakeKaggle()
    original = fake.handler

    def truncated(request):
        resp = original(request)
        if request.url.host == "storage.googleapis.com":
            return httpx.Response(200, content=fake.bundle[:10], headers={"Content-Length": "999"})
        return resp

    kc = KaggleClient(
        access_token="t", client=httpx.Client(transport=httpx.MockTransport(truncated))
    )
    with pytest.raises(KaggleError, match="expected 999"):
        kc.download_all(SLUG, tmp_path / "a.zip")
    assert not (tmp_path / "a.zip").exists() and not (tmp_path / "a.zip.part").exists()


def test_list_submissions_empty_and_paginated():
    assert client_for(FakeKaggle()).list_submissions(SLUG) == []
    subs = [
        {"ref": i, "date": "2026-09-05T10:00:00Z", "status": "COMPLETE", "publicScore": "0.9"}
        for i in range(5)
    ]
    fake = FakeKaggle(submissions=subs, page_size=2)
    got = client_for(fake).list_submissions(SLUG)
    assert [s.ref for s in got] == [0, 1, 2, 3, 4]
    assert got[0].public_score == 0.9 and got[0].complete and not got[0].failed
    assert len([c for c in fake.calls if c.endswith("ListSubmissions")]) == 3


def test_submission_parsing_tolerates_missing_fields():
    fake = FakeKaggle(
        submissions=[
            {"ref": 7},
            {
                "ref": 8,
                "status": "SUBMISSION_STATUS_ERROR",
                "errorDescription": "bad rows",
                "publicScore": "",
            },
        ]
    )
    a, b = client_for(fake).list_submissions(SLUG)
    assert a.status == "PENDING" and a.public_score is None and not a.settled
    assert b.status == "ERROR" and b.failed and b.settled


def test_submit_handshake_and_polling(tmp_path):
    fake = FakeKaggle(upload_503s=1, pending_polls=2, public_score=0.51234)
    file = tmp_path / "submission.csv"
    file.write_text("id,purchased\n1,0.5\n")
    kc = client_for(fake)
    ref = kc.submit(SLUG, file, "hello")
    assert ref == 1001
    assert fake.uploads["tok-1"] == file.read_bytes()
    tail = [c.rsplit("/", 1)[-1] for c in fake.calls]
    assert tail == ["StartSubmissionUpload", "tok-1", "tok-1", "CreateSubmission"], (
        "start, PUT (503), PUT (200), finalise"
    )
    final = kc.wait_for_score(SLUG, ref, timeout=60, interval=1)
    assert final.complete and final.public_score == 0.51234 and final.description == "hello"


def test_submit_failures(tmp_path):
    file = tmp_path / "submission.csv"
    file.write_text("id,purchased\n1,0.5\n")
    fake = FakeKaggle(upload_503s=99)
    with pytest.raises(KaggleError, match="503"):
        client_for(fake).submit(SLUG, file, "x")

    fake = FakeKaggle(fail_submission="Evaluation exception", pending_polls=0)
    kc = client_for(fake)
    ref = kc.submit(SLUG, file, "x")
    final = kc.wait_for_score(SLUG, ref, timeout=60, interval=1)
    assert final.failed and final.error_description == "Evaluation exception"

    fake = FakeKaggle(pending_polls=10**6)
    kc = client_for(fake)
    ref = kc.submit(SLUG, file, "x")
    with pytest.raises(KaggleError, match="still PENDING"):
        kc.wait_for_score(SLUG, ref, timeout=0, interval=1)
    pending = kc.wait_for_score(SLUG, ref, timeout=0, interval=1, raise_on_timeout=False)
    assert pending.ref == ref and not pending.settled and pending.public_score is None
    unlisted = kc.wait_for_score(SLUG, 424242, timeout=0, interval=1, raise_on_timeout=False)
    assert unlisted.ref == 424242 and unlisted.status == "PENDING" and not unlisted.settled


def test_dataset_listing_and_download(tmp_path):
    fake = FakeKaggle(dataset=b"a,b\n1,2\n", dataset_files=["orig.csv"])
    kc = client_for(fake)
    [f] = kc.list_dataset_files("owner/slug")
    assert f.name == "orig.csv" and f.total_bytes == 8
    dest = kc.download_dataset("owner/slug", tmp_path / "d" / "download.bin")
    assert dest.read_bytes() == b"a,b\n1,2\n" and fake.dataset_downloads == 1
    with pytest.raises(KaggleError, match="owner/slug"):
        kc.download_dataset("nope", tmp_path / "x.bin")
    with pytest.raises(KaggleError, match="404"):
        client_for(FakeKaggle()).download_dataset("owner/slug", tmp_path / "y.bin")
    assert not (tmp_path / "y.bin").exists()
