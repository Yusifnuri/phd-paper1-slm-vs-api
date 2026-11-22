import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.final import ledger  # noqa: E402

TEST_HASH = "ab" * 32
CONFIG = {"task": "banking77", "model": "fake", "lr": 0.0004, "n": 200}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


class Env:
    """A temporary git repository with committed final_configs.json and the two arm tags."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.configs = repo / "final_configs.json"
        self.arms = repo / "frozen_arms.json"
        self.ledger = repo / "test_reads.jsonl"
        self.fingerprint = repo / "test_split_fingerprint.json"
        self.calls = 0
        self.config = dict(CONFIG)
        self.test_hash = TEST_HASH

    def kwargs(self, **over):
        kw = dict(task="banking77", repo_root=self.repo, final_configs_path=self.configs,
                  frozen_arms_path=self.arms, ledger_path=self.ledger,
                  fingerprint_path=self.fingerprint, loader=self.loader)
        kw.update(over)
        return kw

    def loader(self):
        self.calls += 1
        return {"texts": ["x"], "hash": TEST_HASH}

    def write_configs(self, arms_hash=None, family="A", config=CONFIG, task="banking77"):
        arms_hash = arms_hash or ledger_hash(self.arms)
        self.configs.write_text(json.dumps({
            "schema": 1, "frozen_arms_sha256": arms_hash,
            "configs": {ledger.config_hash(config): {"task": task, "arm_family": family}}}))

    def commit(self, msg="c"):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", msg)


def ledger_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "config", "tag.gpgsign", "false")
    e = Env(repo)
    e.arms.write_text(json.dumps({"arm": {"lr": 0.0004}}) + "\n")
    e.fingerprint.write_text(json.dumps({
        "banking77": {"n": 1, "sha256": TEST_HASH},
        "billsum": {"n": 1, "sha256": "cd" * 32},
        "ledgar": {"status": "PLACEHOLDER_NOT_FROZEN", "sha256": None}}))
    e.write_configs()
    e.commit("init")
    git(repo, "tag", "prereg-v1")
    git(repo, "tag", "arms-A")
    return e
