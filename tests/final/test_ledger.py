import pytest

from src.final import ledger


def test_config_hash_is_order_independent_and_sensitive():
    assert ledger.config_hash({"a": 1, "b": [1, 2]}) == ledger.config_hash({"b": [1, 2], "a": 1})
    assert ledger.config_hash({"a": 1}) != ledger.config_hash({"a": 2})
    assert len(ledger.config_hash({})) == 64


def test_append_only_lines(tmp_path):
    p = tmp_path / "test_reads.jsonl"
    assert ledger.read_ledger(p) == []
    ledger.append_record(p, {"config_hash": "c", "test_hash": "t", "n": 1})
    first = p.read_bytes()
    ledger.append_record(p, {"config_hash": "c", "test_hash": "t", "n": 2})
    assert p.read_bytes().startswith(first)  # earlier bytes untouched
    assert [r["n"] for r in ledger.read_ledger(p)] == [1, 2]
    assert len(ledger.previous_reads(p, "c", "t")) == 2
    assert ledger.previous_reads(p, "c", "other") == []


def test_refuses_to_append_after_torn_line(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('{"config_hash": "c", "test_hash": "t"}')  # no trailing newline
    with pytest.raises(ledger.LedgerError):
        ledger.append_record(p, {"config_hash": "c", "test_hash": "t"})


def test_malformed_ledger_rejected(tmp_path):
    p = tmp_path / "l.jsonl"
    for bad in ("garbage\n", '{"x": 1}\n', '{"config_hash":"c","test_hash":"t"}\n\n'):
        p.write_text(bad)
        with pytest.raises(ledger.LedgerError):
            ledger.read_ledger(p)


def test_validate_rerun():
    ledger.validate_rerun("crash", "job killed")
    with pytest.raises(ValueError):
        ledger.validate_rerun("worse_than_expected", "x")
    with pytest.raises(ValueError):
        ledger.validate_rerun("api_outage", "  ")
