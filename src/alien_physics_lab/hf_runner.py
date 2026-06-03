from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alien_physics_lab.env import AlienPhysicsLab, LabResult
from alien_physics_lab.openai_runner import _build_prompt, _parse_json_object


@dataclass(frozen=True)
class HFRun:
    result: LabResult
    transcript: list[dict[str, Any]]
    model_messages: list[dict[str, Any]]


@dataclass(frozen=True)
class HFBackend:
    model: Any
    tokenizer: Any
    device: str
    torch: Any


def load_hf_backend(
    *,
    model_id: str,
    device: str = "auto",
    attn_implementation: str = "eager",
) -> HFBackend:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = _resolve_device(device, torch)
    torch_dtype = torch.float16 if resolved_device == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    model.to(resolved_device)
    model.eval()
    return HFBackend(model=model, tokenizer=tokenizer, device=resolved_device, torch=torch)


def run_hf_agent(
    lab: AlienPhysicsLab,
    *,
    backend: HFBackend,
    max_model_turns: int = 10,
    require_calculator: bool = False,
    max_new_tokens: int = 512,
) -> HFRun:
    messages: list[dict[str, Any]] = []
    observations: list[str] = []
    final_answer: dict[str, Any] | None = None

    for turn in range(max_model_turns):
        prompt = _build_prompt(lab, observations)
        response_text = _generate_response(
            model=backend.model,
            tokenizer=backend.tokenizer,
            prompt=prompt,
            device=backend.device,
            torch=backend.torch,
            max_new_tokens=max_new_tokens,
        )
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

        import json

        observations.append(f"Tool observation {lab.tool_calls}: {json.dumps(result, sort_keys=True)}")

        if lab.tool_calls >= lab.max_tool_calls:
            observations.append("Tool budget exhausted. Submit a final answer now.")

    if final_answer is None:
        final_answer = {"gravity_m_s2": None}

    result = lab.score_answer(final_answer)
    return HFRun(result=result, transcript=lab.transcript, model_messages=messages)


def _resolve_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _generate_response(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    torch: Any,
    max_new_tokens: int,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    """Accept a few common local-model tool-call shorthands."""

    if action.get("action") in {"calculator", "drop_ball", "pendulum_period"}:
        tool = str(action["action"])
        args = action.get("args")
        if args is None and tool == "calculator" and "expression" in action:
            args = {"expression": action["expression"]}
        return {"action": "tool", "tool": tool, "args": args or {}}

    return action
