#!/usr/bin/env python
"""Download only the pinned SFT subdirectory and hash weights/processors."""
import argparse
from pathlib import Path

from flowvla_rl.checkpointing import atomic_json
from flowvla_rl.protocol import MODEL_REPO, MODEL_REVISION, MODEL_SUBDIRECTORY
from flowvla_rl.provenance import model_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/smolvla-source")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION,
                             allow_patterns=[f"{MODEL_SUBDIRECTORY}/*"], local_dir=args.output)
    model = Path(root) / MODEL_SUBDIRECTORY
    atomic_json(model / "source_manifest.json", {"repo": MODEL_REPO, "revision": MODEL_REVISION,
                "subdirectory": MODEL_SUBDIRECTORY, "files": model_files(model)})
    print(model.resolve())


if __name__ == "__main__":
    main()
