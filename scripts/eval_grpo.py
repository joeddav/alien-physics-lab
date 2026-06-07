#!/usr/bin/env python
"""Held-out evaluation: run a model through the alien-physics-lab tool loop on a
FIXED set of held-out worlds (fixed seeds) and score it. Lets us compare the base
model vs a GRPO checkpoint on the SAME problems — a confound-free effect size,
unlike per-step training reward (where each step is a different random world).

Reuses AlienPhysicsGRPOEnv so tool dispatch + scoring are identical to training.
Drives the multi-turn ReAct loop over vLLM, BATCHED across all worlds per turn
(apply_chat_template with the env's tool schemas -> generate -> parse <tool_call>
-> dispatch -> repeat) so a 48-world eval finishes in minutes, not hours.

Usage:
    python scripts/eval_grpo.py --model Qwen/Qwen3-1.7B --n 48 --out out/eval-base.json
    python scripts/eval_grpo.py --model out/grpo-soak-lr3e6 --n 48 --out out/eval-soak.json
Both use the same held-out seeds (seed_start=1_000_000), so results are comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import re

SYSTEM = ("You are a careful experimental physicist working in an alien physics lab. "
          "Use the available experiment tools to infer the lab's effective gravity, then "
          "call submit_answer with your best estimate.")
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _ensure_cuda13_runtime() -> None:
    """Preload the CUDA-13 runtime vLLM's _C needs (see scripts/train_grpo.py)."""
    import ctypes
    import glob
    import sysconfig

    libdir = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13", "lib")
    for so in sorted(glob.glob(os.path.join(libdir, "*.so*"))):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _parse_first_tool_call(text: str):
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(1))
        return obj.get("name"), obj.get("arguments", {}) or {}
    except json.JSONDecodeError:
        return None, None


def run_eval(llm, sampling, tokenizer, tool_schemas, seeds, *, thinking, max_turns, noise):
    from alien_physics_lab.grpo_env import AlienPhysicsGRPOEnv, _EpisodeComplete

    envs = [AlienPhysicsGRPOEnv() for _ in seeds]
    convos = []
    for env, s in zip(envs, seeds):
        brief = env.reset(seed=s, measurement_noise=noise, max_tool_calls=5)
        convos.append([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": "Begin the experiment." + brief}])
    done = [False] * len(seeds)
    stats = [{"drop": 0, "pend": 0, "calc": 0, "settings": set(), "repeated": False} for _ in seeds]

    for _turn in range(max_turns):
        active = [i for i in range(len(seeds)) if not done[i]]
        if not active:
            break
        prompts = [tokenizer.apply_chat_template(convos[i], tools=tool_schemas,
                   add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
                   for i in active]
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        for j, i in enumerate(active):
            text = outs[j].outputs[0].text
            if "<tool_call>" in text and "</tool_call>" not in text:
                text += "</tool_call>"
            convos[i].append({"role": "assistant", "content": text})
            name, targs = _parse_first_tool_call(text)
            if name is None:
                done[i] = True
                continue
            if name == "submit_answer":
                try:
                    envs[i].submit_answer(gravity_m_s2=float(targs.get("gravity_m_s2")))
                except _EpisodeComplete:
                    pass
                except (TypeError, ValueError) as exc:
                    convos[i].append({"role": "tool", "content": json.dumps({"error": str(exc)})})
                    continue
                done[i] = True
                continue
            if name == "drop_ball":
                stats[i]["drop"] += 1
            elif name == "pendulum_period":
                stats[i]["pend"] += 1
            elif name == "calculator":
                stats[i]["calc"] += 1
            key = f"{name}:{json.dumps(targs, sort_keys=True)}"
            if key in stats[i]["settings"] and name in ("drop_ball", "pendulum_period"):
                stats[i]["repeated"] = True
            stats[i]["settings"].add(key)
            try:
                result = getattr(envs[i], name)(**targs)
            except Exception as exc:  # noqa: BLE001
                result = json.dumps({"error": str(exc)})
            convos[i].append({"role": "tool", "content": result})

    results = []
    for env, st, s in zip(envs, stats, seeds):
        lr = env.last_result
        pred = lr["pred_g"] if lr else None
        true_g = env._lab.world.effective_gravity_m_s2
        results.append({
            "seed": s, "submitted": bool(env.submitted),
            "score": float(lr["score"]) if lr else 0.0,
            "success": bool(lr["success"]) if lr else False,
            "rel_error": float(lr["relative_error"]) if (lr and lr["relative_error"] is not None) else None,
            "pred_g": pred, "true_g": true_g,
            "n_experiments": st["drop"] + st["pend"], "n_calc": st["calc"],
            "repeated_measurement": st["repeated"],
            "earth_prior_hack": pred is not None and abs(pred - 9.80665) < 0.2 and abs(true_g - 9.80665) > 0.5,
        })
    return results


def summarize(results, meta):
    n = len(results)
    sub = [r for r in results if r["submitted"]]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        **meta, "n": n,
        "submit_rate": len(sub) / n,
        "mean_score": mean([r["score"] for r in results]),
        "success_rate": sum(r["success"] for r in results) / n,
        "mean_score_submitted": mean([r["score"] for r in sub]),
        "success_rate_submitted": (sum(r["success"] for r in sub) / len(sub)) if sub else float("nan"),
        "mean_rel_error_submitted": mean([r["rel_error"] for r in sub]),
        "mean_experiments": mean([r["n_experiments"] for r in results]),
        "frac_ge2_experiments": sum(r["n_experiments"] >= 2 for r in results) / n,
        "repeated_measurement_rate": sum(r["repeated_measurement"] for r in results) / n,
        "earth_prior_hack_rate": sum(r["earth_prior_hack"] for r in results) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed-start", type=int, default=1_000_000)
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--measurement-noise", type=float, default=0.03)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    _ensure_cuda13_runtime()
    from transformers import AutoTokenizer
    from transformers.utils import get_json_schema
    from vllm import LLM, SamplingParams

    from alien_physics_lab.grpo_env import AlienPhysicsGRPOEnv

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    probe = AlienPhysicsGRPOEnv()
    tool_schemas = [get_json_schema(getattr(probe, n)) for n in
                    ("drop_ball", "pendulum_period", "calculator", "submit_answer")]

    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_completion_length + 4096)
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              max_tokens=args.max_completion_length, seed=12345,
                              stop=["</tool_call>"], include_stop_str_in_output=True)

    seeds = [args.seed_start + i for i in range(args.n)]
    results = run_eval(llm, sampling, tokenizer, tool_schemas, seeds,
                       thinking=args.thinking, max_turns=args.max_turns, noise=args.measurement_noise)
    summary = summarize(results, {"model": args.model, "thinking": args.thinking,
                                  "measurement_noise": args.measurement_noise,
                                  "temperature": args.temperature})
    print("=== EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
