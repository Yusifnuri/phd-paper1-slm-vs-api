import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_test_isolation", ROOT / "scripts" / "verify_test_isolation.py")
vti = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vti)


def test_repository_is_clean():
    assert vti.scan(ROOT) == []


def test_detects_split_strings_and_guarded_names(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "good.py").write_text('"""mentions the test split in prose"""\nx = "validation"\n')
    (tmp_path / "src" / "a.py").write_text('from datasets import load_dataset\nds = load_dataset("d", split="test")\n')
    (tmp_path / "src" / "b.py").write_text("ds = load('d', split='test[:5]')\n")
    (tmp_path / "src" / "c.py").write_text("from src.train.billsum import load_us_test_subset_for_final_eval\n")
    (tmp_path / "src" / "d.py").write_text("import m\nm.load_us_test_subset_for_final_eval()\nm._US_TEST_SPLIT\n")
    (tmp_path / "src" / "e.py").write_text("ds = load('d', split='ca_test')\n")
    found = vti.scan(tmp_path, allowed=set(), dirs=("src",))
    files = {f.split(":")[0] for f in found}
    assert files == {"src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py"}


def test_allowed_readers_are_exempt(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "reader.py").write_text("ds = load('d', split='test')\n")
    assert vti.scan(tmp_path, allowed={"src/reader.py"}, dirs=("src",)) == []
    assert vti.scan(tmp_path, allowed=set(), dirs=("src",)) != []


def test_allowed_set_is_exactly_the_reviewed_readers():
    assert vti.ALLOWED_READERS == {
        "src/train/billsum.py", "src/final/final_eval.py", "scripts/freeze_test_split.py",
        "scripts/freeze_test_splits.py", "scripts/verify_test_isolation.py"}
    for rel in vti.ALLOWED_READERS:
        assert (ROOT / rel).exists(), rel


def test_final_package_only_final_eval_reads_test():
    # ledger.py and __init__ must stay free of test-split references
    for name in ("ledger.py", "__init__.py"):
        assert vti.violations_in_source((ROOT / "src" / "final" / name).read_text()) == []
