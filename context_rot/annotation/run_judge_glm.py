#!/usr/bin/env python3
"""Run the E1 judge annotation with GLM 5.3 (z.ai coding endpoint) as the judge.

    JUDGE_BUNDLE_DIR=/path/to/judge_bundle python3 run_judge_glm.py --concurrency 16

JUDGE_BUNDLE_DIR (default: cwd) must hold MANIFEST.json, glm_key.json
({"zai_key": ...}, never committed) and windows/shard_*.json; results/ is written there.

Adapted from the bundle's run_judge.py for the z.ai API:
  * endpoint  https://api.z.ai/api/coding/paas/v4 (key in glm_key.json)
  * model     glm-5.3, thinking enabled, reasoning_effort=high (manifest says xhigh;
              GLM offers low/high/max - "high" chosen for the full run)
  * z.ai does not enforce response_format json_schema (no guided decoding), so the
    frozen role schema is appended to the user message after the transcript and the
    reply is validated locally against it; invalid replies are retried (3 draws,
    per MANIFEST judge_model.retry). The frozen system message and transcript text
    are unchanged.
  * quota-aware: on an explicit balance/quota 429 the run pauses and probes every
    5 min until quota returns. Rate-limit 429s (code 1302) are NOT quota: workers
    just retry them with capped backoff, which keeps the pipe full.
  * resumable per WINDOW, not just per shard: each finished window is appended to
    results/partial_shard_NNN.jsonl; the final results_shard_NNN.json is written
    when a shard completes. Stop and restart freely.

Settings kept from MANIFEST.json: temperature 0.0, max_tokens 32768 (do NOT
raise), 3 draws per window, failed windows returned with parser_failed=true.
"""
from __future__ import annotations

import argparse, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

HERE = Path(os.environ.get("JUDGE_BUNDLE_DIR", ".")).resolve()   # the judge bundle dir
M = json.loads((HERE / "MANIFEST.json").read_text())
ROLE, JM = M["role"], M["judge_model"]
KEY = json.loads((HERE / "glm_key.json").read_text())["zai_key"]

BASE_URL = "https://api.z.ai/api/coding/paas/v4"
MODEL = "glm-5.3"
REASONING_EFFORT = "high"         # user's choice 2026-08-20 (manifest says xhigh; first
                                  # 114 windows ran at "max" - see run_metadata.json)
SCHEMA = ROLE["schema"]

QUOTA_PROBE_SECONDS = 300         # while quota-paused, probe every 5 min so a fresh
                                  # quota window starts being used within minutes
TRANSPORT_RETRIES = 12            # per draw, for 5xx / network errors / short rate limits

OUTPUT_INSTRUCTION = (
    "\n\nOUTPUT FORMAT (mandatory): Respond with a single JSON object and nothing else - "
    "no markdown fences, no prose before or after. The object must contain exactly the "
    "keys required by this JSON Schema and validate against it:\n"
    + json.dumps(SCHEMA, indent=1)
)

FAMILIES = {"W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9"}


def validate_payload(p: Any) -> str:
    """Return '' if p conforms to the frozen role schema, else a reason."""
    if not isinstance(p, dict):
        return "payload is not an object"
    req = set(SCHEMA["required"])
    got = set(p.keys())
    if req - got:
        return f"missing keys: {sorted(req - got)}"
    if got - req:
        return f"extra keys: {sorted(got - req)}"
    def is_int(v):  # bool is an int subclass in Python; the schema means real integers
        return isinstance(v, int) and not isinstance(v, bool)
    for k in ("opportunity_present", "rejection_present", "adoption_present",
              "requires_human_review"):
        if not isinstance(p[k], bool):
            return f"{k} is not a boolean"
    for k in ("first_opportunity_turn", "first_rejection_turn", "adoption_turn"):
        v = p[k]
        if v is not None and (not is_int(v) or v < 0):
            return f"{k} is not a non-negative integer or null"
    for k in ("opportunity_family", "rejection_family", "adoption_family"):
        v = p[k]
        if v is not None and v not in FAMILIES:
            return f"{k} not in W1..W9 or null"
    if p["rejection_type"] not in {"normative", "instrumental", "mixed", "unclear", "none"}:
        return "rejection_type not in allowed labels"
    for k in ("rejection_evidence", "adoption_evidence"):
        if not isinstance(p[k], str):
            return f"{k} is not a string"
    v = p["adoption_window"]
    if v is not None:
        if (not isinstance(v, list) or len(v) != 2
                or not all(is_int(x) and x >= 0 for x in v)):
            return "adoption_window is not [int, int] with non-negative ints, or null"
    v = p["adoption_mode"]
    if v is not None and v not in {"verbal_commitment", "behavioral"}:
        return "adoption_mode not in allowed labels"
    c = p["confidence"]
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0.0 <= c <= 1.0):
        return "confidence not a number in [0,1]"
    return ""


def extract_json(content: str) -> Any:
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON object found", text[:80], 0)


class QuotaGate:
    """When quota is exhausted, one thread probes until it returns; the rest wait."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._paused = False
        self.pauses = 0

    def _probe(self, client: httpx.Client) -> bool:
        try:
            r = client.post(f"{BASE_URL}/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Reply with the word ok."}],
                "temperature": 0.0, "max_tokens": 256,
                "thinking": {"type": "enabled"}, "reasoning_effort": "low",
            }, headers={"Authorization": f"Bearer {KEY}"}, timeout=120)
            return r.status_code == 200
        except Exception:
            return False

    def pause_and_wait(self, client: httpx.Client) -> None:
        with self._cond:
            if self._paused:
                while self._paused:
                    self._cond.wait()
                return
            self._paused = True
            self.pauses += 1
        print(f"[quota] exhausted at {time.strftime('%H:%M:%S')} - pausing, "
              f"probing every {QUOTA_PROBE_SECONDS // 60} min", flush=True)
        while True:
            time.sleep(QUOTA_PROBE_SECONDS)
            if self._probe(client):
                break
            print(f"[quota] still exhausted at {time.strftime('%H:%M:%S')}", flush=True)
        with self._cond:
            self._paused = False
            self._cond.notify_all()
        print(f"[quota] restored at {time.strftime('%H:%M:%S')} - resuming", flush=True)


GATE = QuotaGate()


def is_quota_error(status: int, body_text: str) -> bool:
    """Only explicit balance/quota signals count as quota exhaustion.

    z.ai's coding endpoint returns 429 code 1302 "Rate limit reached for requests"
    for concurrency bursts - that is a throughput cap, never a reason to pause;
    workers just retry it with capped backoff.
    """
    if status != 429:
        return False
    t = body_text.lower()
    return any(s in t for s in ("balance", "recharge", "quota", "package",
                                "insufficient", "1113"))


def post_once(client: httpx.Client, body: dict[str, Any]) -> dict[str, Any]:
    """One judge draw. Retries transport trouble; rate-limit 429s retry indefinitely
    with short backoff, escalating to a quota pause only if they persist."""
    backoff = 30.0                # for 5xx / network errors
    rl_backoff = 3.0              # for burst 429s (code 1302) - these clear in seconds
    transport_left = TRANSPORT_RETRIES
    while True:
        try:
            r = client.post(f"{BASE_URL}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {KEY}"})
        except httpx.HTTPError:
            transport_left -= 1
            if transport_left <= 0:
                raise RuntimeError("transport retries exhausted (network errors)")
            time.sleep(backoff + random.uniform(0, 10))
            backoff = min(backoff * 2, 300)
            continue
        if r.status_code == 200:
            return r.json()
        if is_quota_error(r.status_code, r.text):
            print(f"[quota] trigger: {r.text[:160]}", flush=True)
            GATE.pause_and_wait(client)
            continue
        if r.status_code == 429:
            # Rate-limit 429 (code 1302): a throughput cap, not quota. Never pause -
            # rejected requests cost no tokens, and retrying keeps the pipe full the
            # moment a slot frees. (A probe-gate here parks workers while capacity
            # exists: observed 403 windows completing DURING a 4 h "pause".)
            time.sleep(rl_backoff + random.uniform(0, 3))
            rl_backoff = min(rl_backoff * 2, 30)
            continue
        if r.status_code in (500, 502, 503, 504):
            transport_left -= 1
            if transport_left <= 0:
                raise RuntimeError(f"transport retries exhausted (HTTP {r.status_code})")
            time.sleep(backoff + random.uniform(0, 10))
            backoff = min(backoff * 2, 300)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")


def judge_one(client: httpx.Client, transcript: str) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": ROLE["system_message"]},
            {"role": "user",
             "content": ROLE["input_template"].format(TRANSCRIPT=transcript)
                        + OUTPUT_INSTRUCTION},
        ],
        "temperature": JM["temperature"],
        "max_tokens": JM["max_tokens"],           # do NOT raise - see MANIFEST judge_model
        "thinking": {"type": "enabled"},
        "reasoning_effort": REASONING_EFFORT,
        "response_format": {"type": "json_object"},
    }
    reasoning_chars, err = 0, ""
    for attempt in range(1, 4):                   # 3 draws: a retry is an independent sample
        try:
            data = post_once(client, body)
            msg = data["choices"][0]["message"]
            reasoning_chars = len(msg.get("reasoning_content") or msg.get("reasoning") or "")
            payload = extract_json(msg.get("content") or "")
            why = validate_payload(payload)
            if why:
                raise ValueError(f"schema: {why}")
            return {"payload": payload, "parser_failed": False, "attempts": attempt,
                    "reasoning_chars": reasoning_chars, "error": ""}
        except Exception as exc:                  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
    return {"payload": {}, "parser_failed": True, "attempts": 3,
            "reasoning_chars": reasoning_chars, "error": err}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N windows this run (0 = no limit), for smoke tests")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta_path = args.out / "run_metadata.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "judge_model": MODEL,
            "endpoint": BASE_URL,
            "thinking": "enabled",
            "reasoning_effort": REASONING_EFFORT,
            "temperature": JM["temperature"],
            "max_tokens": JM["max_tokens"],
            "deviations_from_manifest": [
                "Judge is glm-5.3 via api.z.ai, not thinkingmachines/Inkling-Small-NVFP4: "
                "the bundled judge model was not available; annotations are a GLM-5.3 read "
                "of the same frozen windows/prompts.",
                "reasoning_effort 'max' (GLM's ceiling) in place of 'xhigh'.",
                "z.ai does not enforce response_format json_schema, so the frozen role "
                "schema is appended verbatim to the user message after the transcript and "
                "replies are validated locally against it (invalid -> retry, 3 draws). "
                "System message and transcript text are byte-identical to the manifest.",
            ],
            "system_message_sha256": ROLE["system_message_sha256"],
            "schema_sha256": ROLE["schema_sha256"],
        }, indent=1))

    shards = sorted((HERE / "windows").glob("shard_*.json"))
    print(f"{len(shards)} shards, {M['n_windows']:,} windows total", flush=True)
    started = time.monotonic()
    done_this_run = 0
    stop = False

    with httpx.Client(timeout=1800.0) as client:
        for sp in shards:
            if stop:
                break
            dest = args.out / f"results_{sp.stem}.json"
            partial = args.out / f"partial_{sp.stem}.jsonl"
            if dest.exists():
                print(f"  {sp.name}: already done, skipping", flush=True)
                continue
            shard = json.loads(sp.read_text())

            done: dict[str, dict[str, Any]] = {}
            if partial.exists():
                for line in partial.read_text().splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        done[rec["window_id"]] = rec
            todo = [w for w in shard if w["window_id"] not in done]
            if args.limit:
                todo = todo[:max(0, args.limit - done_this_run)]
            print(f"  {sp.name}: {len(shard)} windows, {len(done)} already done, "
                  f"{len(todo)} to go", flush=True)

            write_lock = threading.Lock()
            counter = {"n": 0, "failed": 0}

            def one(w: dict[str, Any]) -> None:
                r = judge_one(client, w["transcript"])
                r["window_id"] = w["window_id"]
                with write_lock:
                    with partial.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    done[w["window_id"]] = r
                    counter["n"] += 1
                    if r["parser_failed"]:
                        counter["failed"] += 1
                        print(f"    [fail] {w['window_id']}: {r['error'][:160]}", flush=True)
                    if counter["n"] % 25 == 0:
                        el = (time.monotonic() - started) / 3600
                        print(f"    {sp.name}: {counter['n']}/{len(todo)} this run, "
                              f"{counter['failed']} unparseable, "
                              f"{(done_this_run + counter['n']) / el:.0f} windows/h",
                              flush=True)

            t0 = time.monotonic()
            if todo:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    list(pool.map(one, todo))
            done_this_run += counter["n"]

            if len(done) == len(shard):
                results = [done[w["window_id"]] for w in shard]   # shard order
                dest.write_text(json.dumps(results, indent=1))
                failed = sum(1 for r in results if r["parser_failed"])
                mins = (time.monotonic() - t0) / 60
                print(f"  {sp.name}: complete - {len(results)} windows "
                      f"({mins:.0f} min this run), {failed} unparseable total -> {dest.name}",
                      flush=True)
            else:
                print(f"  {sp.name}: stopped at {len(done)}/{len(shard)} "
                      f"(resume by re-running)", flush=True)
            if args.limit and done_this_run >= args.limit:
                stop = True

    print(f"\ndone -> {args.out}\nSend the whole `results/` folder back.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
