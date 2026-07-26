#!/usr/bin/env python3
"""Check or publish the GitHub model-card SSOT to Hugging Face.

Hugging Face renders ``README.md`` as the model card. This repository keeps the
same content in ``MODEL-CARD.md`` so research, provenance, and release changes
are reviewed with the public code. This tool prevents the two copies from
quietly drifting.

Examples:

    python tools/sync_hf_model_card.py --check
    python tools/sync_hf_model_card.py --publish

API publishing uses the normal Hugging Face credential lookup (``HF_TOKEN`` or
``hf auth login``). Git publishing uses the normal Git credential manager. No
token is read or printed by this script.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_REPO_ID = "weemed/IlhaEmbed"
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "MODEL-CARD.md"


def raw_url(repo_id: str, revision: str) -> str:
    return f"https://huggingface.co/{repo_id}/raw/{revision}/README.md"


def source_text() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    required = (
        "---\n",
        "license: apache-2.0",
        "library_name: sentence-transformers",
        "pipeline_tag: sentence-similarity",
        "# IlhaEmbed",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise ValueError(f"{SOURCE} is missing required model-card metadata: {missing}")
    return text


def remote_text(repo_id: str, revision: str) -> str:
    request = urllib.request.Request(
        raw_url(repo_id, revision),
        headers={"User-Agent": "ilhaembed-model-card-sync/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def check(repo_id: str, revision: str) -> bool:
    local = source_text()
    remote = remote_text(repo_id, revision)
    if local == remote:
        print(f"model card in sync: {repo_id}@{revision}")
        return True

    print(f"model card drift: {SOURCE} != {repo_id}@{revision}/README.md", file=sys.stderr)
    diff = difflib.unified_diff(
        remote.splitlines(),
        local.splitlines(),
        fromfile=f"{repo_id}@{revision}/README.md",
        tofile=str(SOURCE.relative_to(ROOT)),
        lineterm="",
        n=3,
    )
    for line in list(diff)[:120]:
        print(line, file=sys.stderr)
    return False


def publish_api(repo_id: str) -> None:
    source_text()
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "publishing requires huggingface-hub; install requirements-publish.txt"
        ) from error

    api = HfApi()
    result = api.upload_file(
        path_or_fileobj=SOURCE,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: sync model card from WeeMed/ilhaembed",
    )
    print(f"published model card: {result}")


def publish_git(repo_id: str) -> None:
    """Publish through the normal Git credential manager without reading a token."""
    with tempfile.TemporaryDirectory(prefix="ilhaembed-hf-card-") as tmp:
        checkout = Path(tmp) / "model"
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://huggingface.co/{repo_id}", str(checkout)],
            check=True,
        )
        destination = checkout / "README.md"
        if destination.exists() and destination.read_bytes() == SOURCE.read_bytes():
            print(f"model card already current in git checkout: {repo_id}")
            return
        shutil.copyfile(SOURCE, destination)
        subprocess.run(["git", "-C", str(checkout), "add", "README.md"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "commit",
                "-m",
                "docs: sync model card from WeeMed/ilhaembed",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "push", "origin", "main"], check=True)
        print(f"published model card through git: {repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="compare local SSOT with the HF card")
    action.add_argument("--publish", action="store_true", help="upload local SSOT, then verify it")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--transport",
        choices=("api", "git"),
        default="api",
        help="publishing transport; git uses the normal Git credential manager",
    )
    args = parser.parse_args()

    if args.publish:
        if args.transport == "git":
            publish_git(args.repo_id)
        else:
            publish_api(args.repo_id)
        return 0 if check(args.repo_id, args.revision) else 1
    return 0 if check(args.repo_id, args.revision) else 1


if __name__ == "__main__":
    raise SystemExit(main())
