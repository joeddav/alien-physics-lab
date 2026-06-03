from __future__ import annotations

from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.request
from typing import Any

from alien_physics_lab.env import AlienPhysicsLab, LabResult

RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass(frozen=True)
class OpenAIRun:
    result: LabResult
    transcript: list[dict[str, Any]]
    model_messages: list[dict[str, Any]]


def run_openai_agent(
    lab: AlienPhysicsLab,
    *,
    model: str = "gpt-5-nano",
    max_model_turns: int = 10,
    require_calculator: bool = False,
    reasoning_effort: str = "minimal",
    request_timeout_s: int = 300,
    api_key: str | None = None,
) -> OpenAIRun:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    messages: list[dict[str, Any]] = []
    observations: list[str] = []
    final_answer: dict[str, Any] | None = None

    for turn in range(max_model_turns):
        prompt = _build_prompt(lab, observations)
        response_text = _responses_json(
            api_key=api_key,
            model=model,
            prompt=prompt,
            reasoning_effort=reasoning_effort,
            request_timeout_s=request_timeout_s,
        )
        action = _parse_json_object(response_text)
        messages.append({"turn": turn, "raw": response_text, "parsed": action})

        if not action:
            observations.append("Your last response was not valid JSON. Try again.")
            continue

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
    return OpenAIRun(result=result, transcript=lab.transcript, model_messages=messages)


def _build_prompt(lab: AlienPhysicsLab, observations: list[str]) -> str:
    history = "\n".join(observations) if observations else "No experiments have been run yet."
    return (
        f"{lab.instructions()}\n\n"
        "The lab's gravity is not necessarily Earth gravity. Do not guess 9.81 "
        "unless your observations support it. If you have no observations yet, "
        "run one experiment first.\n\n"
        "Useful formulas:\n"
        "- drop_ball gives height h and measured fall time t, so g = 2h / t^2.\n"
        "- pendulum_period gives length L and measured period T, so g = 4*pi^2*L / T^2.\n"
        "Use the numeric observations from this episode to estimate gravity. "
        "The calculator tool is available if you want arithmetic help. Never put "
        "arithmetic expressions in final JSON; return a finished decimal number. "
        "Calculator input must be one arithmetic expression only: no variables, "
        "assignments, or semicolons.\n\n"
        "Worked example: if drop_ball returns height_m=10 and measured_time_s=2, "
        "then g=20/4=5, so "
        'reply {"action":"final","gravity_m_s2":5.0}.\n\n'
        "Reply with exactly one JSON object and no surrounding text.\n"
        "To run a tool:\n"
        '{"action":"tool","tool":"drop_ball","args":{"mass_kg":1,"height_m":10}}\n'
        "or:\n"
        '{"action":"tool","tool":"pendulum_period","args":{"length_m":5}}\n'
        "or:\n"
        '{"action":"tool","tool":"calculator","args":{"expression":"2*10/(1.5**2)"}}\n'
        "To finish:\n"
        '{"action":"final","gravity_m_s2":14.715}\n\n'
        f"Observations so far:\n{history}"
    )


def _responses_json(
    *,
    api_key: str,
    model: str,
    prompt: str,
    reasoning_effort: str,
    request_timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 8_000 if reasoning_effort == "high" else 2_000,
        "reasoning": {"effort": reasoning_effort},
        "text": {"verbosity": "low"},
    }
    request = urllib.request.Request(
        RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=request_timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc

    text = _extract_output_text(data)
    if text is None:
        raise RuntimeError(f"OpenAI response did not contain output text: {data}")
    return text


def _extract_output_text(data: dict[str, Any]) -> str | None:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]

    return None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None
