# purple-terminal-bench

A purple (attacker-side) agent for the [AgentBeats](https://agentbeats.org) **Coding Agent** track, evaluated on [Terminal-Bench 2.0](https://www.tbench.ai). The agent receives terminal-based engineering tasks over A2A, drives a real Linux shell through the `terminal-bench-shell-v1` protocol (multi-turn `exec_request` / `exec_result`), and emits a final answer when the task is solved.

## Abstract

This agent implements **Recursive Language Models (RLM)** — the inference strategy introduced by Zhang, Khattab, and Kraska (MIT CSAIL, 2025) in [_Recursive Language Models_](https://arxiv.org/abs/2512.24601). RLM lets a root LM decompose long-context work by handing chunks to a recursive `llm_query()` call inside a Python interpreter, instead of cramming everything into one window. We adopt that pattern wholesale and wrap it around the Terminal-Bench shell protocol.

A root agent (`claude-opus-4-7`) runs a ReAct loop with three tools:

- **`bash`** — execute a command in the task's shell via the green-side `exec_request`. Full stdout/stderr is captured; the chat sees a truncated preview.
- **`repl`** — a persistent Python interpreter where a `context` list accumulates **untruncated** records of every bash command, output, and prior repl execution. The model can grep, slice, and summarize that history without re-paying the token cost of pulling full outputs into its own window.
- **`final`** — emit the answer for the task.

Inside the REPL the model has **`llm_query(prompt: str) -> str`** — this is the RLM recursive call. It dispatches to a fast sub-LLM (`claude-haiku-4-5`) with a ~400K-char input budget. The root model uses it to offload bulk-context work — "scan this 5K-line log for the failure", "summarize this man page", "extract the failing assertion from this trace" — without burning its own context window.

Recursion is bounded (`MAX_TOTAL_BASH=60`, `MAX_TOTAL_REPL=60`, `MAX_TOTAL_LLM_QUERY=30`, `MAX_INNER_STEPS_PER_TURN=12`).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  A2A server (a2a-python)                                     │
│  ── multi-turn: task → exec_request → exec_result → final    │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │  Root LLM        │   claude-opus-4-7
                        │  (ReAct loop)    │   tool-required
                        └─────┬────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
   ┌─────────┐          ┌──────────┐           ┌──────────┐
   │  bash   │          │   repl   │           │  final   │
   │  exec_  │          │ persist. │           │ deliver  │
   │ request │          │  python  │           │  answer  │
   └─────────┘          └─────┬────┘           └──────────┘
                              │
                              ▼
                       context: list[dict]   ◄── full untruncated
                              │                  bash/repl history
                              ▼
                       llm_query(prompt) ──► claude-haiku-4-5
                                            (~400K char sub-LLM)
```

The green agent owns the shell; we only ever request command execution. The Recursive-LM scratchpad lives entirely on our side and is what lets a single Opus head stretch across many shell turns without exhausting its window.

## Project structure

```
src/
├─ server.py        # A2A server + agent card (Terminal-Bench skill)
├─ executor.py      # A2A request handling
├─ agent.py         # RLM-style ReAct loop (root + REPL + sub-LLM)
└─ messenger.py     # A2A messaging utilities
amber-manifest.json5
Dockerfile
```

## Running locally

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run src/server.py
```

## Submission

Deployed via Amber manifest (`amber-manifest.json5`) and submitted through the AgentBeats Terminal-Bench 2.0 track. The Amber image is built from `Dockerfile` and pushed to `ghcr.io/zaidishahbaz1/purple-terminal-bench:latest` by GitHub Actions on push to `main`.

## Citation

If you build on this agent, please also cite the original paper:

```bibtex
@article{zhang2025recursive,
  title   = {Recursive Language Models},
  author  = {Zhang, Alex and Khattab, Omar and Kraska, Tim},
  journal = {arXiv preprint arXiv:2512.24601},
  year    = {2025},
  institution = {MIT CSAIL},
  url     = {https://arxiv.org/abs/2512.24601},
}
```

## Acknowledgments

Built on the [RDI-Foundation/agent-template](https://github.com/RDI-Foundation/agent-template). Evaluated on [Terminal-Bench 2.0](https://www.tbench.ai). Architecture inspired by [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Khattab, Kraska — MIT CSAIL, 2025).
