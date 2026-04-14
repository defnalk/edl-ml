"""CLI integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edl_ml.cli.main import main


def test_simulate_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "sim.json"
    rc = main(
        [
            "simulate",
            "--concentration",
            "0.1",
            "--e-min",
            "-0.2",
            "--e-max",
            "0.2",
            "--n-points",
            "11",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert "data" in payload
    assert len(payload["data"]) == 11
    for row in payload["data"]:
        assert row["capacitance_uf_cm2"] > 0


def test_generate_produces_parquet(tmp_path: Path) -> None:
    out = tmp_path / "tiny.parquet"
    rc = main(
        [
            "generate",
            "--n-samples",
            "4",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_unknown_command_errors() -> None:
    with pytest.raises(SystemExit):
        main(["nope"])
