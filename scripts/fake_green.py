"""Fake Green Agent for local end-to-end testing.

Drives the terminal-bench-shell-v1 protocol against a running Purple Agent:
- Sends an initial `task` message with an instruction
- Executes any `exec_request` via subprocess in a working dir (default: tempdir)
- Replies with `exec_result`
- Stops on `final`

WARNING: runs arbitrary bash commands the LLM emits. Default cwd is an ephemeral
tempdir, but bash can still escape it. Run only on your dev machine.
"""
import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from messenger import Messenger  # noqa: E402


def run_bash(cmd: str, cwd: Path, timeout: int) -> dict:
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "kind": "exec_result",
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
        return {
            "kind": "exec_result",
            "exit_code": -1,
            "stdout": stdout or "",
            "stderr": f"[timeout after {timeout}s]",
        }


def _short(s: str, n: int = 800) -> str:
    if len(s) <= n:
        return s
    return f"{s[:n]}\n... [truncated {len(s) - n} chars]"


async def drive(url: str, instruction: str, cwd: Path, max_turns: int):
    messenger = Messenger()
    payload: dict = {
        "kind": "task",
        "protocol": "terminal-bench-shell-v1",
        "instruction": instruction,
    }
    new_conv = True
    for turn in range(max_turns):
        out = json.dumps(payload)
        print(f"\n--- turn {turn} :: green -> purple ---\n{_short(out)}")
        resp = await messenger.talk_to_agent(out, url, new_conversation=new_conv, timeout=600)
        new_conv = False
        print(f"\n--- turn {turn} :: purple -> green ---\n{_short(resp, 2000)}")
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            print("[fake green] non-JSON response, stopping")
            return
        kind = data.get("kind")
        if kind == "final":
            print(f"\n=== FINAL ===\n{data.get('output', '')}")
            print(f"\n[fake green] cwd was: {cwd}")
            return
        if kind == "exec_request":
            cmd = data.get("command", "")
            tmo = int(data.get("timeout", 30))
            print(f"\n[fake green] $ {cmd}  (cwd={cwd}, timeout={tmo}s)")
            payload = run_bash(cmd, cwd, tmo)
            print(
                f"[fake green] exit={payload['exit_code']} "
                f"stdout_len={len(payload['stdout'])} stderr_len={len(payload['stderr'])}"
            )
            continue
        print(f"[fake green] unknown kind: {kind}, stopping")
        return
    print(f"\n=== max turns ({max_turns}) reached ===")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:9010")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--cwd", default=None, help="working dir for bash; default: fresh tempdir")
    ap.add_argument("--max-turns", type=int, default=30)
    args = ap.parse_args()

    print("[fake green] WARNING: runs arbitrary LLM-emitted bash. Use a safe dev box.\n")

    if args.cwd:
        cwd = Path(args.cwd).expanduser().resolve()
        cwd.mkdir(parents=True, exist_ok=True)
        asyncio.run(drive(args.url, args.instruction, cwd, args.max_turns))
    else:
        with tempfile.TemporaryDirectory(prefix="fake-green-") as td:
            cwd = Path(td)
            print(f"[fake green] using tempdir: {cwd}")
            asyncio.run(drive(args.url, args.instruction, cwd, args.max_turns))


if __name__ == "__main__":
    main()
