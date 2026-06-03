from __future__ import annotations

import argparse
import json
import statistics

from alien_physics_lab.agents import result_to_dict, run_heuristic_agent
from alien_physics_lab.env import AlienPhysicsLab
from alien_physics_lab.hf_runner import load_hf_backend, run_hf_agent
from alien_physics_lab.mlx_runner import load_mlx_backend, run_mlx_agent
from alien_physics_lab.openai_runner import run_openai_agent
from alien_physics_lab.world import EARTH_GRAVITY_M_S2, WorldParams


def main() -> None:
    parser = argparse.ArgumentParser(description="Alien physics lab playground")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="Run one scripted heuristic episode")
    _add_lab_args(play)

    heuristic = subparsers.add_parser("eval-heuristic", help="Evaluate the heuristic agent")
    _add_lab_args(heuristic)
    heuristic.add_argument("--episodes", type=int, default=20)

    openai = subparsers.add_parser("eval-openai", help="Evaluate an OpenAI model")
    _add_lab_args(openai)
    openai.add_argument("--episodes", type=int, default=5)
    openai.add_argument("--model", default="gpt-5-nano")
    openai.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="minimal",
    )
    openai.add_argument("--request-timeout-s", type=int, default=300)
    openai.add_argument("--require-calculator", action="store_true")
    openai.add_argument("--show-transcripts", action="store_true")

    hf = subparsers.add_parser("eval-hf", help="Evaluate a local Hugging Face model")
    _add_lab_args(hf)
    hf.add_argument("--episodes", type=int, default=5)
    hf.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    hf.add_argument("--device", default="auto")
    hf.add_argument("--attn-implementation", default="eager")
    hf.add_argument("--max-new-tokens", type=int, default=512)
    hf.add_argument("--max-model-turns", type=int, default=10)
    hf.add_argument("--require-calculator", action="store_true")
    hf.add_argument("--show-transcripts", action="store_true")

    mlx = subparsers.add_parser("eval-mlx", help="Evaluate a local MLX model")
    _add_lab_args(mlx)
    mlx.add_argument("--episodes", type=int, default=5)
    mlx.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    mlx.add_argument("--max-tokens", type=int, default=96)
    mlx.add_argument("--max-model-turns", type=int, default=10)
    mlx.add_argument("--require-calculator", action="store_true")
    mlx.add_argument("--show-transcripts", action="store_true")

    args = parser.parse_args()

    if args.command == "play":
        lab = _make_lab(args.seed, args)
        result = run_heuristic_agent(lab)
        print(json.dumps({"result": result_to_dict(result), "transcript": lab.transcript}, indent=2))
    elif args.command == "eval-heuristic":
        _eval_heuristic(args)
    elif args.command == "eval-openai":
        _eval_openai(args)
    elif args.command == "eval-hf":
        _eval_hf(args)
    elif args.command == "eval-mlx":
        _eval_mlx(args)


def _add_lab_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gravity", type=float, default=None, help="Fixed effective gravity in m/s^2")
    parser.add_argument("--gravity-min", type=float, default=0.5 * EARTH_GRAVITY_M_S2)
    parser.add_argument("--gravity-max", type=float, default=2.0 * EARTH_GRAVITY_M_S2)
    parser.add_argument("--measurement-noise", type=float, default=0.005)
    parser.add_argument("--max-tool-calls", type=int, default=5)


def _make_lab(seed: int, args: argparse.Namespace) -> AlienPhysicsLab:
    if args.gravity is None:
        return AlienPhysicsLab.random_gravity_lab(
            seed=seed,
            gravity_min=args.gravity_min,
            gravity_max=args.gravity_max,
            measurement_noise=args.measurement_noise,
            max_tool_calls=args.max_tool_calls,
        )

    return AlienPhysicsLab(
        world=WorldParams(
            gravity_m_s2=args.gravity,
            measurement_noise=args.measurement_noise,
            seed=seed,
        ),
        max_tool_calls=args.max_tool_calls,
    )


def _eval_heuristic(args: argparse.Namespace) -> None:
    results = []
    for i in range(args.episodes):
        lab = _make_lab(args.seed + i, args)
        results.append(run_heuristic_agent(lab))

    print(json.dumps(_summarize(results), indent=2))


def _eval_openai(args: argparse.Namespace) -> None:
    results = []
    runs = []
    for i in range(args.episodes):
        lab = _make_lab(args.seed + i, args)
        run = run_openai_agent(
            lab,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            request_timeout_s=args.request_timeout_s,
            require_calculator=args.require_calculator,
        )
        results.append(run.result)
        if args.show_transcripts:
            runs.append(
                {
                    "episode": i,
                    "result": result_to_dict(run.result),
                    "lab_transcript": run.transcript,
                    "model_messages": run.model_messages,
                }
            )

    output = {"summary": _summarize(results)}
    if args.show_transcripts:
        output["runs"] = runs
    print(json.dumps(output, indent=2))


def _eval_hf(args: argparse.Namespace) -> None:
    backend = load_hf_backend(
        model_id=args.model,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    results = []
    runs = []
    for i in range(args.episodes):
        lab = _make_lab(args.seed + i, args)
        run = run_hf_agent(
            lab,
            backend=backend,
            max_new_tokens=args.max_new_tokens,
            max_model_turns=args.max_model_turns,
            require_calculator=args.require_calculator,
        )
        results.append(run.result)
        if args.show_transcripts:
            runs.append(
                {
                    "episode": i,
                    "result": result_to_dict(run.result),
                    "lab_transcript": run.transcript,
                    "model_messages": run.model_messages,
                }
            )

    output = {"summary": _summarize(results)}
    if args.show_transcripts:
        output["runs"] = runs
    print(json.dumps(output, indent=2))


def _eval_mlx(args: argparse.Namespace) -> None:
    backend = load_mlx_backend(model_id=args.model)
    results = []
    runs = []
    for i in range(args.episodes):
        lab = _make_lab(args.seed + i, args)
        run = run_mlx_agent(
            lab,
            backend=backend,
            max_tokens=args.max_tokens,
            max_model_turns=args.max_model_turns,
            require_calculator=args.require_calculator,
        )
        results.append(run.result)
        if args.show_transcripts:
            runs.append(
                {
                    "episode": i,
                    "result": result_to_dict(run.result),
                    "lab_transcript": run.transcript,
                    "model_messages": run.model_messages,
                }
            )

    output = {"summary": _summarize(results)}
    if args.show_transcripts:
        output["runs"] = runs
    print(json.dumps(output, indent=2))


def _summarize(results) -> dict[str, object]:
    scores = [result.score for result in results]
    rel_errors = [result.relative_error for result in results if result.relative_error is not None]
    tool_calls = [result.tool_calls for result in results]
    calculator_calls = [result.calculator_calls for result in results]
    return {
        "episodes": len(results),
        "success_rate": sum(result.success for result in results) / len(results),
        "mean_score": statistics.fmean(scores),
        "mean_relative_error": statistics.fmean(rel_errors) if rel_errors else None,
        "mean_tool_calls": statistics.fmean(tool_calls),
        "mean_calculator_calls": statistics.fmean(calculator_calls),
        "results": [result_to_dict(result) for result in results],
    }


if __name__ == "__main__":
    main()
