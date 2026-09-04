#!/usr/bin/env python3
"""Watch $JUDGE_BUNDLE_DIR/results/ and upload each completed shard to the HF dataset
thebajajra/muse-trajectory-annotations under the `commit-hook` subset.

Converts results_shard_NNN.json (a JSON list) to JSONL at
commit-hook/results_shard_NNN.jsonl. The remote repo listing is the source of
truth for what is already uploaded. Exits once all 8 shards are up.
"""
from __future__ import annotations

import json, os, time
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(os.environ.get("JUDGE_BUNDLE_DIR", ".")).resolve()   # the judge bundle dir
RESULTS = HERE / "results"
REPO = "thebajajra/muse-trajectory-annotations"
SUBSET = "commit-hook"
N_SHARDS = 8
CHECK_SECONDS = 600

README = """---
configs:
  - config_name: commit-hook
    data_files:
      - split: train
        path: commit-hook/results_shard_*.jsonl
---

# MUSE trajectory annotations

Judge annotations of coding-agent trajectories.

## Subset: `commit-hook`

Event-sequence annotations of 7,593 transcript windows drawn from 433 complete
trajectories of a coding agent working on a git pre-commit-hook task (E1). For
each window the judge identifies the earliest concrete workaround opportunity,
the earliest rejection of a workaround (labelled normative / instrumental /
mixed / unclear), and the earliest later adoption, with quoted evidence.

- **Judge**: `glm-5.3` (z.ai), thinking enabled, temperature 0.0. Most windows
  annotated at `reasoning_effort: high`; the first 114 at `max` (ids listed in
  `commit-hook/run_metadata.json`).
- **Record shape**: `window_id` (joins back to the source trajectories),
  `payload` (labels conforming to the frozen role schema), `parser_failed`,
  `attempts` (1-3 draws), `reasoning_chars`, `error`. Windows that never parsed
  are kept with `parser_failed: true` so denominators stay honest.
- **Families** W1-W9 refer to the study's workaround taxonomy; turn indices are
  absolute turn numbers in the source trajectory.
- One file per shard: `commit-hook/results_shard_000.jsonl` ... `_007.jsonl`,
  ~1,000 windows each. `run_metadata.json` records judge settings and the
  deviations from the study manifest.
"""


def main() -> int:
    api = HfApi()
    while True:
        remote = set(api.list_repo_files(REPO, repo_type="dataset"))
        ops = 0

        if "README.md" not in remote:
            api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md",
                            repo_id=REPO, repo_type="dataset",
                            commit_message="Add commit-hook subset config")
            print(f"[{time.strftime('%H:%M:%S')}] uploaded README.md", flush=True)
            ops += 1

        meta = RESULTS / "run_metadata.json"
        if meta.exists() and f"{SUBSET}/run_metadata.json" not in remote:
            api.upload_file(path_or_fileobj=meta, path_in_repo=f"{SUBSET}/run_metadata.json",
                            repo_id=REPO, repo_type="dataset",
                            commit_message="Add commit-hook run metadata")
            print(f"[{time.strftime('%H:%M:%S')}] uploaded run_metadata.json", flush=True)
            ops += 1

        n_up = 0
        for i in range(N_SHARDS):
            local = RESULTS / f"results_shard_{i:03d}.json"
            dest = f"{SUBSET}/results_shard_{i:03d}.jsonl"
            if dest in remote:
                n_up += 1
                continue
            if not local.exists():
                continue
            recs = json.loads(local.read_text())
            jsonl = "".join(json.dumps(r) + "\n" for r in recs)
            api.upload_file(path_or_fileobj=jsonl.encode(), path_in_repo=dest,
                            repo_id=REPO, repo_type="dataset",
                            commit_message=f"Add shard {i:03d} ({len(recs)} windows)")
            print(f"[{time.strftime('%H:%M:%S')}] uploaded {dest} ({len(recs)} windows)",
                  flush=True)
            n_up += 1
            ops += 1

        if n_up == N_SHARDS:
            print(f"[{time.strftime('%H:%M:%S')}] all {N_SHARDS} shards uploaded - done",
                  flush=True)
            return 0
        if not ops:
            pass  # nothing new this cycle
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
