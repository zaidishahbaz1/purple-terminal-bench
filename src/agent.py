import asyncio
import contextlib
import io
import json
import logging
import os
import re
import traceback
from typing import Any

import anthropic
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

logger = logging.getLogger(__name__)

ROOT_MODEL = os.environ.get("ROOT_MODEL", "claude-opus-4-7")
SUB_MODEL = os.environ.get("SUB_MODEL", "claude-haiku-4-5-20251001")
ROOT_MAX_TOKENS = 4096
SUB_MAX_TOKENS = 4096

MAX_INNER_STEPS_PER_TURN = 12
MAX_TOTAL_BASH = 60
MAX_TOTAL_REPL = 60
MAX_TOTAL_LLM_QUERY = 30
LLM_OBS_TRUNCATE = 6000
SUB_PROMPT_MAX_CHARS = 400_000

SYSTEM_PROMPT = """You are an autonomous agent solving a task in a Linux terminal. Each step you must call exactly ONE of three tools: bash, repl, or final.

WORKING MEMORY MODEL (Recursive LM):
You have a `context` Python variable (a list, persistent across all repl calls) that records every bash command + output and every repl execution. The full untruncated outputs live there. The chat you see only contains TRUNCATED previews of bash/repl outputs to save tokens. When you need to inspect a full bash output, look it up via repl, e.g.:

    print(context[-1]['stdout'][:5000])
    [print(c.get('command')) for c in context if c['kind']=='bash']

You also have llm_query(prompt: str) -> str inside the repl: a fast helper LLM (Haiku, ~400K chars input). Use it to:
- summarize huge outputs you found in context
- classify or extract over thousands of items
- run line-by-line semantic transforms over chunks

A viable pattern when an output is too large to scan in chat: call repl, slice the relevant part out of context, pass it to llm_query with a precise question, get a condensed answer.

TOOLS:
- bash(command, timeout=30): run a shell command in the task env. Output auto-truncated to ~6KB in chat; full version in context[-1]. timeout clamped to [1,300]s.
- repl(code): in-process Python over `context` and `llm_query`. NEVER use repl to run shell commands (use bash). Use repl ONLY for context inspection, slicing, and llm_query.
- final(output): call ONLY when the task is complete. Pass a brief description of what you did.

DISCIPLINE:
- One tool per response.
- Don't re-read the same files. Cache findings as repl variables.
- Verify your changes work before declaring victory — score is based on automated tests over the env.
- Read the task instructions carefully and address every requirement.
- For any output >2KB you actually need to understand, prefer llm_query over re-scrolling chat.
- When confident the task is solved, call final. Don't stall.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": (
            "Run a bash command in the task environment. Output (stdout, stderr, exit_code) "
            "is appended to `context` and a truncated view is returned. "
            "Use this for any shell action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
                "timeout": {
                    "type": "integer",
                    "description": "Seconds; clamped to [1,300]. Default 30.",
                    "minimum": 1,
                    "maximum": 300,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "repl",
        "description": (
            "Execute Python in a persistent in-process REPL. Globals: `context` (running "
            "transcript list of dicts), `llm_query(prompt)` (Haiku sub-LLM, ~400K chars). "
            "DO NOT run shell commands here — use bash."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "final",
        "description": "Terminate the task with a final result message.",
        "input_schema": {
            "type": "object",
            "properties": {"output": {"type": "string"}},
            "required": ["output"],
        },
    },
]


def _truncate(s: str, n: int = LLM_OBS_TRUNCATE) -> str:
    if not s:
        return s
    if len(s) <= n:
        return s
    half = n // 2 - 50
    return f"{s[:half]}\n... [TRUNCATED {len(s) - 2 * half} chars; full in context[-1]] ...\n{s[-half:]}"


class Agent:
    """One Agent per A2A context_id (one task). RLM-flavored loop:
    root LM (Opus) drives bash/repl/final tool calls; repl exposes a persistent
    `context` variable and an `llm_query` sub-LM (Haiku).
    """

    def __init__(self) -> None:
        self.messenger = Messenger()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._has_key = bool(api_key)
        if self._has_key:
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
            self.sync_client = anthropic.Anthropic(api_key=api_key)
        self.history: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []
        self.repl_globals: dict[str, Any] = {}
        self.repl_initialized = False
        self.bash_count = 0
        self.repl_count = 0
        self.llm_query_count = 0
        self.instruction: str | None = None
        self.pending_bash_command: str | None = None
        self.pending_bash_tool_id: str | None = None
        self.pending_extra_tool_results: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    def _init_repl(self) -> None:
        if self.repl_initialized:
            return

        sync_client = self.sync_client

        def llm_query(prompt: str) -> str:
            self.llm_query_count += 1
            if self.llm_query_count > MAX_TOTAL_LLM_QUERY:
                return "[llm_query budget exhausted]"
            text = prompt[:SUB_PROMPT_MAX_CHARS]
            try:
                resp = sync_client.messages.create(
                    model=SUB_MODEL,
                    max_tokens=SUB_MAX_TOKENS,
                    messages=[{"role": "user", "content": text}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception as e:
                return f"[llm_query error: {e}]"

        self.repl_globals = {
            "context": self.transcript,
            "llm_query": llm_query,
            "json": json,
            "re": re,
        }
        self.repl_initialized = True

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        async with self._lock:
            await self._run_locked(message, updater)

    async def _run_locked(self, message: Message, updater: TaskUpdater) -> None:
        if not self._has_key:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=json.dumps({"kind": "final", "output": "[ANTHROPIC_API_KEY not set]"})))],
                name="Result",
            )
            return

        input_text = get_message_text(message)
        try:
            payload = json.loads(input_text)
            if not isinstance(payload, dict):
                payload = {"kind": "task", "instruction": input_text}
        except json.JSONDecodeError:
            payload = {"kind": "task", "instruction": input_text}

        kind = payload.get("kind", "task")

        if kind == "task" or self.instruction is None:
            self.instruction = payload.get("instruction") or payload.get("prompt") or input_text
            self._init_repl()
            self.history = [
                {
                    "role": "user",
                    "content": (
                        f"Task instruction:\n\n{self.instruction}\n\n"
                        "Begin. Call exactly one of bash, repl, or final."
                    ),
                }
            ]
        elif kind == "exec_result" and self.pending_bash_tool_id is not None:
            entry = {
                "kind": "bash",
                "command": self.pending_bash_command or "",
                "exit_code": payload.get("exit_code"),
                "stdout": payload.get("stdout", "") or "",
                "stderr": payload.get("stderr", "") or "",
            }
            self.transcript.append(entry)
            self.repl_globals["context"] = self.transcript
            obs = (
                f"exit_code={entry['exit_code']}\n"
                f"stdout (truncated):\n{_truncate(entry['stdout'])}\n"
                f"stderr (truncated):\n{_truncate(entry['stderr'])}"
            )
            content_list: list[dict[str, Any]] = [
                {"type": "tool_result", "tool_use_id": self.pending_bash_tool_id, "content": obs}
            ]
            content_list.extend(self.pending_extra_tool_results)
            self.pending_extra_tool_results = []
            self.history.append({"role": "user", "content": content_list})
            self.pending_bash_command = None
            self.pending_bash_tool_id = None
        else:
            self.history.append(
                {
                    "role": "user",
                    "content": f"Received message of kind={kind}. Continue or call final.",
                }
            )

        outbound = await self._inner_loop(updater)
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=outbound))],
            name="Result",
        )

    async def _inner_loop(self, updater: TaskUpdater) -> str:
        for step in range(MAX_INNER_STEPS_PER_TURN):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"step {step}: thinking"),
            )

            try:
                resp = await self.async_client.messages.create(
                    model=ROOT_MODEL,
                    max_tokens=ROOT_MAX_TOKENS,
                    system=[
                        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
                    ],
                    tools=TOOLS,
                    messages=self.history,
                )
            except Exception as e:
                logger.exception("root LM error")
                return json.dumps({"kind": "final", "output": f"agent root LM error: {e}"})

            self.history.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})

            tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
            if not tool_use_blocks:
                self.history.append(
                    {
                        "role": "user",
                        "content": "You did not call a tool. Call exactly one of bash, repl, or final.",
                    }
                )
                continue

            primary = tool_use_blocks[0]
            extras = tool_use_blocks[1:]
            extra_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": "[ignored: call exactly one tool per response]",
                    "is_error": True,
                }
                for b in extras
            ]

            name = primary.name
            args = primary.input or {}
            tool_id = primary.id

            if name == "final":
                return json.dumps({"kind": "final", "output": str(args.get("output", ""))})

            if name == "bash":
                if self.bash_count >= MAX_TOTAL_BASH:
                    self.history.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": "[bash budget exhausted; call final]",
                                    "is_error": True,
                                },
                                *extra_results,
                            ],
                        }
                    )
                    continue
                cmd = str(args.get("command", "")).strip()
                if not cmd:
                    self.history.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": "[empty command]",
                                    "is_error": True,
                                },
                                *extra_results,
                            ],
                        }
                    )
                    continue
                tmo = args.get("timeout", 30)
                try:
                    tmo = max(1, min(int(tmo), 300))
                except (TypeError, ValueError):
                    tmo = 30
                self.bash_count += 1
                self.pending_bash_command = cmd
                self.pending_bash_tool_id = tool_id
                self.pending_extra_tool_results = extra_results
                return json.dumps({"kind": "exec_request", "command": cmd, "timeout": tmo})

            if name == "repl":
                if self.repl_count >= MAX_TOTAL_REPL:
                    self.history.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": "[repl budget exhausted]",
                                    "is_error": True,
                                },
                                *extra_results,
                            ],
                        }
                    )
                    continue
                code = str(args.get("code", ""))
                self.repl_count += 1
                stdout, stderr, exc = await asyncio.to_thread(self._exec_repl, code)
                obs_parts = []
                if stdout:
                    obs_parts.append(f"stdout:\n{_truncate(stdout)}")
                if stderr:
                    obs_parts.append(f"stderr:\n{_truncate(stderr)}")
                if exc:
                    obs_parts.append(f"exception:\n{exc}")
                obs = "\n\n".join(obs_parts) or "(no output)"
                self.transcript.append(
                    {"kind": "repl", "code": code, "stdout": stdout, "stderr": stderr, "exception": exc}
                )
                self.history.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": tool_id, "content": obs},
                            *extra_results,
                        ],
                    }
                )
                continue

            self.history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": f"[unknown tool {name}]",
                            "is_error": True,
                        },
                        *extra_results,
                    ],
                }
            )

        return json.dumps({"kind": "final", "output": "[step cap reached without final]"})

    def _exec_repl(self, code: str) -> tuple[str, str, str | None]:
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        exc: str | None = None
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, self.repl_globals)
        except Exception:
            exc = traceback.format_exc()
        return stdout_buf.getvalue(), stderr_buf.getvalue(), exc
