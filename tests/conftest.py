from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

MINIMAL_CONFIG = {
    "competition": {"slug": "playground-series-s6e9"},
    "assessor": {"enabled": False},  # interviews are tested explicitly
    "backends": {
        "local": {"base_url": "http://local.test/v1", "model": "m-local"},
        "nvidia": {
            "base_url": "https://nv.test/v1/",
            "model": "m-nv",
            "api_key": "${NV_KEY}",
            "max_concurrency": 4,
        },
    },
}


@pytest.fixture
def write_config(tmp_path):
    def _write(data: dict, name: str = "c.yaml") -> Path:
        p = tmp_path / name
        p.write_text(yaml.safe_dump(data))
        return p

    return _write


@pytest.fixture
def minimal_config():
    import copy

    return copy.deepcopy(MINIMAL_CONFIG)
