"""Manual E2E smoke for SGLANG_FLASHINFER_VERIFY_ATTN_MODE.

Launches a real EAGLE3 + FlashInfer server once per mode (legacy / prepacked /
split / chain), sends identical greedy requests (sequential + concurrent to
exercise both cuda-graph replay and eager verify), and checks:
- "prepacked" produces exactly the same outputs as "legacy" (same fa2 kernel,
  identical mask bits);
- "split" produces coherent outputs with a comparable accept length (its
  merge_state numerics may diverge token-for-token under greedy sampling);
- "chain" (requires topk=1, so it is compared against a dedicated topk=1
  legacy baseline) produces exactly the same outputs as that baseline
  (MaskMode.CAUSAL is bit-identical to the causal-chain custom mask on fa2).

Run manually (needs a local Qwen3-8B target + EAGLE3 draft):
  python3 test/manual/e2e_flashinfer_verify_attn_modes.py \
      --target /path/to/Qwen3-8B --draft /path/to/Qwen3-8B_eagle3
"""

import argparse
import concurrent.futures
import json

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

PROMPTS = [
    "The capital of France is",
    "Write a haiku about mountains and rivers.",
    "Explain the difference between a list and a tuple in Python.",
    "1, 1, 2, 3, 5, 8, 13. The next three numbers of this sequence are",
    "Translate to German: 'The weather is nice today.'",
    "def quicksort(arr):",
]


def run_mode(mode: str, args, topk: int = 4) -> dict:
    base_url = f"http://127.0.0.1:{args.port}"
    with envs.SGLANG_FLASHINFER_VERIFY_ATTN_MODE.override(mode):
        process = popen_launch_server(
            args.target,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm", "EAGLE3",
                "--speculative-draft-model-path", args.draft,
                "--speculative-num-steps", "7" if topk == 1 else "3",
                "--speculative-eagle-topk", str(topk),
                "--speculative-num-draft-tokens", "8",
                "--attention-backend", "flashinfer",
                "--mem-fraction-static", str(args.mem_fraction),
                "--cuda-graph-max-bs", "4",
                "--max-running-requests", "8",
            ],
        )
    try:
        def generate(prompt):
            r = requests.post(
                base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 64,
                    },
                },
                timeout=300,
            )
            r.raise_for_status()
            return r.json()

        # Sequential (bs=1 -> cuda-graph replay path).
        seq_results = [generate(p) for p in PROMPTS[:3]]
        # Concurrent burst (bs up to 6 > cuda-graph-max-bs=4 -> also exercises
        # the eager verify path and graph padding).
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            conc_results = list(pool.map(generate, PROMPTS))

        info = requests.get(base_url + "/get_server_info", timeout=60).json()
        internal = info.get("internal_states", [{}])
        accept_len = (internal[0] if internal else {}).get("avg_spec_accept_length")
        return {
            "mode": mode,
            "sequential": [r["text"] for r in seq_results],
            "concurrent": [r["text"] for r in conc_results],
            "avg_spec_accept_length": accept_len,
        }
    finally:
        kill_process_tree(process.pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--mem-fraction", type=float, default=0.4)
    parser.add_argument("--modes", default="legacy,prepacked,split")
    args = parser.parse_args()

    results = {}
    for mode in args.modes.split(","):
        if mode == "chain":
            # chain requires topk=1; pair it with a dedicated topk=1 legacy
            # baseline for an exact token-for-token comparison.
            for key, env_mode in (("legacy_topk1", "legacy"), ("chain", "chain")):
                print(f"===== running mode: {key} =====", flush=True)
                results[key] = run_mode(env_mode, args, topk=1)
                print(
                    json.dumps(results[key], indent=2, ensure_ascii=False),
                    flush=True,
                )
            continue
        print(f"===== running mode: {mode} =====", flush=True)
        results[mode] = run_mode(mode, args)
        print(json.dumps(results[mode], indent=2, ensure_ascii=False), flush=True)

    failures = []
    if "legacy" in results and "prepacked" in results:
        for key in ("sequential", "concurrent"):
            if results["legacy"][key] != results["prepacked"][key]:
                for a, b in zip(results["legacy"][key], results["prepacked"][key]):
                    if a != b:
                        failures.append(
                            f"prepacked {key} output mismatch:\n legacy: {a!r}\n prepacked: {b!r}"
                        )
    if "legacy" in results and "split" in results:
        la = results["legacy"]["avg_spec_accept_length"]
        sa = results["split"]["avg_spec_accept_length"]
        if la and sa and abs(sa - la) / la > 0.25:
            failures.append(f"split accept length {sa} deviates >25% from legacy {la}")
        for a, b in zip(
            results["legacy"]["sequential"], results["split"]["sequential"]
        ):
            # Greedy outputs may eventually diverge from numerics; require a
            # substantial common prefix as a sanity floor.
            common = len([1 for x, y in zip(a, b) if x == y])
            if common < min(len(a), len(b)) * 0.5 and a[:32] != b[:32]:
                failures.append(f"split output diverges early:\n legacy: {a!r}\n split: {b!r}")

    if "legacy_topk1" in results and "chain" in results:
        for key in ("sequential", "concurrent"):
            if results["legacy_topk1"][key] != results["chain"][key]:
                for a, b in zip(results["legacy_topk1"][key], results["chain"][key]):
                    if a != b:
                        failures.append(
                            f"chain {key} output mismatch:\n legacy_topk1: {a!r}\n chain: {b!r}"
                        )
        la = results["legacy_topk1"]["avg_spec_accept_length"]
        ca = results["chain"]["avg_spec_accept_length"]
        if la and ca and abs(ca - la) / la > 0.02:
            failures.append(
                f"chain accept length {ca} deviates from topk=1 legacy {la}"
            )

    accepts = {m: r["avg_spec_accept_length"] for m, r in results.items()}
    print("accept lengths:", accepts)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("-", f)
        raise SystemExit(1)
    print("\nE2E smoke passed for modes:", ", ".join(results))


if __name__ == "__main__":
    main()
