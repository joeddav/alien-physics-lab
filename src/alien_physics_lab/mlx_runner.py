from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from alien_physics_lab.env import AlienPhysicsLab, LabResult
from alien_physics_lab.hf_runner import _normalize_action
from alien_physics_lab.openai_runner import _build_prompt, _parse_json_object


@dataclass(frozen=True)
class MLXBackend:
    model: Any
    tokenizer: Any


@dataclass(frozen=True)
class MLXRun:
    result: LabResult
    transcript: list[dict[str, Any]]
    model_messages: list[dict[str, Any]]


def load_mlx_backend(*, model_id: str) -> MLXBackend:
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    model_path = snapshot_download(model_id, local_files_only=True)
    model, tokenizer = load(model_path)
    return MLXBackend(model=model, tokenizer=tokenizer)


def run_mlx_agent(
    lab: AlienPhysicsLab,
    *,
    backend: MLXBackend,
    max_model_turns: int = 10,
    require_calculator: bool = False,
    max_tokens: int = 96,
) -> MLXRun:
    from mlx_lm import generate

    messages: list[dict[str, Any]] = []
    observations: list[str] = []
    final_answer: dict[str, Any] | None = None

    for turn in range(max_model_turns):
        prompt = _build_prompt(lab, observations)
        prompt_text = _chat_prompt(backend.tokenizer, prompt)
        response_text = generate(
            backend.model,
            backend.tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            verbose=False,
        ).strip()
        action = _parse_json_object(response_text)
        messages.append({"turn": turn, "raw": response_text, "parsed": action})

        if not action:
            observations.append("Your last response was not valid JSON. Try again.")
            continue

        action = _normalize_action(action)
        is_final = action.get("action") == "final" or (
            "gravity_m_s2" in action and action.get("action") is None
        )
        if is_final:
            if require_calculator and lab.tool_calls > 0 and lab.calculator_calls == 0:
                observations.append(
                    "Final answer rejected: use the calculator tool with the "
                    "latest observation before submitting a final answer."
                )
                continue
            final_answer = action
            break

        if action.get("action") != "tool":
            observations.append('Invalid action. Use "tool" or "final".')
            continue

        tool = action.get("tool")
        args = action.get("args", {})
        if not isinstance(tool, str) or not isinstance(args, dict):
            observations.append("Invalid tool call. Provide string tool and object args.")
            continue

        try:
            result = lab.call_tool(tool, **args)
        except (TypeError, ValueError) as exc:
            result = {"error": str(exc)}

        observations.append(f"Tool observation {lab.tool_calls}: {json.dumps(result, sort_keys=True)}")

        if lab.tool_calls >= lab.max_tool_calls:
            observations.append("Tool budget exhausted. Submit a final answer now.")

    if final_answer is None:
        final_answer = {"gravity_m_s2": None}

    result = lab.score_answer(final_answer)
    return MLXRun(result=result, transcript=lab.transcript, model_messages=messages)


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "apply_chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
