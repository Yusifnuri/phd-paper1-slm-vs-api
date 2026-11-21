"""Push adapters to a private Hugging Face Hub repo. The token comes from HF_TOKEN or the
`huggingface-cli login` cache (huggingface_hub reads both); it is never logged or stored."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def push_folder(repo_id: str, folder: str, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=folder, path_in_repo=path_in_repo,
        commit_message=f"Add {path_in_repo}",
    )


def push_file(repo_id: str, file: str, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    if not Path(file).exists():
        return
    HfApi().upload_file(
        repo_id=repo_id, repo_type="model", path_or_fileobj=file, path_in_repo=path_in_repo,
        commit_message=f"Update {path_in_repo}",
    )


def safe_error(e: BaseException, limit: int = 200) -> str:
    """Short error text for the results file (class name plus a truncated message)."""
    return f"{type(e).__name__}: {str(e)[:limit]}"
