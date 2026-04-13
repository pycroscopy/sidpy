#!/usr/bin/env python3
"""
Simple MCP client for testing sidpy's BEPS MCP server with a local Ollama model.

This script can:
1. Launch the sidpy BEPS MCP server over stdio
2. List the exposed MCP tools
3. Call a tool directly with JSON arguments
4. Run a lightweight Ollama-driven tool loop against the MCP server
5. Generate small synthetic BEPS / SHO datasets for quick server testing

Examples
--------
List tools:
    python3 examples/mcp/run_beps_mcp_client.py list-tools

Direct BEPS test with synthetic data:
    python3 examples/mcp/run_beps_mcp_client.py demo-beps

Direct SHO test with synthetic data:
    python3 examples/mcp/run_beps_mcp_client.py demo-sho

Chat through Ollama:
    python3 examples/mcp/run_beps_mcp_client.py chat \
        --model llama3.1 \
        --prompt "List the tools and run a small BEPS fit test"

Call a tool manually:
    python3 examples/mcp/run_beps_mcp_client.py call \
        --tool fit_beps_loops_tool \
        --args-json '{"data":[[0,1,0,-1]],"dc_voltage":[-1,0,1,2]}'
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_SERVER_COMMAND = (
    f"{shlex.quote(sys.executable)} -c "
    "\"from sidpy.proc.mcp_server_beps import main; main()\""
)
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
MCP_PROTOCOL_VERSION = "2024-11-05"


class McpError(RuntimeError):
    """Raised when the MCP server returns an error or the transport breaks."""


@dataclass
class JsonRpcMessage:
    """In-memory representation of a single JSON-RPC message."""

    payload: Dict[str, Any]


def _json_default(value: Any) -> Any:
    """Encode values that the standard JSON encoder cannot handle."""
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


class StdioMcpClient:
    """Small MCP client that speaks JSON-RPC over stdio."""

    def __init__(self, server_command: str, cwd: Optional[Path] = None, debug: bool = False):
        self.server_command = server_command
        self.cwd = str(cwd) if cwd else None
        self.proc: Optional[subprocess.Popen] = None
        self._next_id = 1
        self.debug = debug

    def __enter__(self) -> "StdioMcpClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self.proc is not None:
            return

        self.proc = subprocess.Popen(
            shlex.split(self.server_command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
        )
        self._debug(f"started server: {self.server_command}")
        time.sleep(0.1)
        if self.proc.poll() is not None:
            stderr_text = ""
            if self.proc.stderr is not None:
                stderr_text = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise McpError(f"MCP server exited during startup.\n{stderr_text}".strip())
        self.initialize()

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        finally:
            self.proc = None

    def initialize(self) -> Dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "sidpy-ollama-mcp-client",
                    "version": "0.1.0",
                },
            },
        )
        self.notify("notifications/initialized", {})
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self.request("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._debug(f"notify -> {method}")
        self._send(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }
        )

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._debug(f"request #{request_id} -> {method}")
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )

        while True:
            message = self._read_message()
            payload = message.payload
            if "id" not in payload:
                continue
            if payload["id"] != request_id:
                continue
            if "error" in payload:
                raise McpError(json.dumps(payload["error"], indent=2))
            self._debug(f"response #{request_id} <- {method}")
            return payload.get("result", {})

    def _send(self, payload: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise McpError("MCP server is not running.")
        body = (json.dumps(payload, default=_json_default) + "\n").encode("utf-8")
        self._debug(f"send: {payload.get('method', 'response')}")
        self.proc.stdin.write(body)
        self.proc.stdin.flush()

    def _read_message(self) -> JsonRpcMessage:
        if self.proc is None or self.proc.stdout is None:
            raise McpError("MCP server is not running.")
        line = self.proc.stdout.readline()
        if not line:
            stderr_text = ""
            if self.proc.stderr is not None:
                stderr_text = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise McpError(f"MCP server closed the stream.\n{stderr_text}".strip())
        decoded = line.decode("utf-8", errors="replace").strip()
        self._debug(f"recv: {decoded[:200]}")
        return JsonRpcMessage(json.loads(decoded))

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[debug] {message}", file=sys.stderr, flush=True)


def _ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    ollama_url: str,
    timeout: int,
    format_json: bool = False,
    stream: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if format_json:
        payload["format"] = "json"

    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not stream:
                return json.loads(resp.read().decode("utf-8"))

            assembled_content: List[str] = []
            final_obj: Optional[Dict[str, Any]] = None
            while True:
                line = resp.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                chunk = json.loads(text)
                message = chunk.get("message", {})
                piece = message.get("content", "")
                if piece:
                    assembled_content.append(piece)
                    print(piece, end="", file=sys.stderr, flush=True)
                if chunk.get("done"):
                    final_obj = chunk
                    break

            print("", file=sys.stderr, flush=True)
            if final_obj is None:
                raise RuntimeError("Ollama stream ended before a final response was received.")

            final_message = dict(final_obj.get("message", {}))
            final_message["content"] = "".join(assembled_content)
            final_obj["message"] = final_message
            return final_obj
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {ollama_url}. "
            "Make sure `ollama serve` is running."
        ) from exc


def _timed_call(label: str, func, *args, **kwargs):
    """Run a callable and report elapsed time."""
    start = time.time()
    print(f"[progress] {label}...", file=sys.stderr, flush=True)
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"[progress] {label} finished in {elapsed:.1f}s", file=sys.stderr, flush=True)
    return result


def _extract_tool_text(result: Dict[str, Any]) -> str:
    content = result.get("content", [])
    chunks = []
    for item in content:
        if item.get("type") == "text":
            chunks.append(item.get("text", ""))
    if chunks:
        return "\n".join(chunks)
    return json.dumps(result, indent=2)


def _coerce_json_like(value: Any) -> Any:
    """Convert model-produced JSON-looking strings into real Python values."""
    if isinstance(value, dict):
        return {key: _coerce_json_like(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_coerce_json_like(item) for item in value]
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    if text[0] in "[{":
        try:
            return _coerce_json_like(json.loads(text))
        except json.JSONDecodeError:
            return value

    if text in {"null", "true", "false"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    return value


def _summarize_value(value: Any) -> Any:
    """Summarize large nested payloads for terminal-friendly logging."""
    if isinstance(value, dict):
        return {key: _summarize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        if not value:
            return []
        if all(not isinstance(item, (list, dict)) for item in value):
            if len(value) <= 8:
                return value
            return {
                "type": "list",
                "length": len(value),
                "preview": value[:3] + ["..."] + value[-2:],
            }
        shape = []
        current = value
        while isinstance(current, list) and current:
            shape.append(len(current))
            current = current[0]
        return {"type": "nested_list", "shape": shape}
    return value


def _try_parse_json_text(text: str) -> Any:
    """Parse text as JSON when possible, otherwise return the original string."""
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


def _preview_nested_numbers(value: Any, depth: int = 0) -> Any:
    """Return a small preview from nested numeric lists."""
    if depth >= 2:
        return value
    if isinstance(value, list) and value:
        if all(not isinstance(item, list) for item in value):
            return value[:4]
        return _preview_nested_numbers(value[0], depth + 1)
    return value


def _summarize_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact summary from the MCP tool result payload."""
    text = _extract_tool_text(result)
    parsed = _try_parse_json_text(text)

    if not isinstance(parsed, dict):
        return {"text_preview": text[:400]}

    summary: Dict[str, Any] = {}
    for key in ("fit_kind", "parameter_labels", "parameter_shape", "covariance_shape"):
        if key in parsed:
            summary[key] = parsed[key]

    parameters = parsed.get("parameters")
    if parameters is not None:
        summary["parameter_preview"] = _preview_nested_numbers(parameters)

    metadata = parsed.get("parameter_metadata")
    if isinstance(metadata, dict):
        compact_metadata = {}
        for key in ("fit_parms_dict", "fitting_functions", "guess_function", "source_code"):
            if key in metadata:
                compact_metadata[key] = _summarize_value(metadata[key])
        if compact_metadata:
            summary["parameter_metadata"] = compact_metadata

    if not summary:
        summary["text_preview"] = text[:400]
    return summary


def build_demo_beps_payload() -> Dict[str, Any]:
    """Generate a tiny synthetic BEPS dataset for a quick end-to-end fit."""
    import numpy as np

    from sidpy.proc.mcp_server_beps import loop_fit_function

    rng = np.random.default_rng(7)
    dc_voltage = np.linspace(-4.0, 4.0, 64)
    shape = (2, 2, dc_voltage.size)
    data = np.zeros(shape, dtype=float)

    base_params = np.array([-0.4, 1.8, -1.2, 1.3, 0.03, 2.0, 2.2, 2.0, 2.1])
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            params = base_params.copy()
            params[0] += 0.1 * ix
            params[1] += 0.08 * iy
            params[2] -= 0.05 * iy
            params[3] += 0.05 * ix
            noise = rng.normal(0.0, 0.02, size=dc_voltage.size)
            data[ix, iy, :] = loop_fit_function(dc_voltage, *params) + noise

    return {
        "data": data.tolist(),
        "dc_voltage": dc_voltage.tolist(),
        "use_kmeans": False,
        "return_cov": False,
        "loss": "linear",
        "dataset_name": "synthetic_beps_demo",
    }


def build_demo_sho_payload() -> Dict[str, Any]:
    """Generate a tiny synthetic SHO dataset for a quick end-to-end fit."""
    import numpy as np

    from sidpy.proc.mcp_server_beps import SHO_fit_flattened

    rng = np.random.default_rng(11)
    freq = np.linspace(300_000.0, 340_000.0, 96)
    shape = (2, 2, freq.size)
    complex_data = np.zeros(shape, dtype=np.complex128)

    base_params = np.array([0.03, 320_000.0, 60.0, 0.2])
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            params = base_params.copy()
            params[0] *= 1.0 + 0.05 * ix
            params[1] += 500.0 * iy
            params[2] += 5.0 * ix
            params[3] += 0.02 * iy
            flattened = SHO_fit_flattened(freq, *params)
            half = len(flattened) // 2
            curve = flattened[:half] + 1j * flattened[half:]
            noise = rng.normal(0.0, 5e-4, size=freq.size) + 1j * rng.normal(0.0, 5e-4, size=freq.size)
            complex_data[ix, iy, :] = curve + noise

    return {
        "real_data": np.real(complex_data).tolist(),
        "imag_data": np.imag(complex_data).tolist(),
        "frequency": freq.tolist(),
        "use_kmeans": False,
        "return_cov": False,
        "loss": "linear",
        "dataset_name": "synthetic_sho_demo",
    }


def _pretty_tool_summary(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for tool in tools:
        name = tool.get("name", "<unknown>")
        description = tool.get("description", "").strip()
        lines.append(f"- {name}")
        if description:
            lines.append(f"  {description}")
    return "\n".join(lines)


def run_ollama_tool_loop(
    client: StdioMcpClient,
    model: str,
    prompt: str,
    ollama_url: str,
    timeout: int,
    max_steps: int,
    ollama_stream: bool,
) -> str:
    """Minimal JSON-based tool loop that works with Ollama chat responses."""
    tools = client.list_tools()
    tool_calls_made = 0
    tool_catalog = []
    for tool in tools:
        tool_catalog.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "inputSchema": tool.get("inputSchema", {}),
            }
        )

    demo_beps_payload = build_demo_beps_payload()
    demo_sho_payload = build_demo_sho_payload()

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are testing an MCP server. "
                "Return valid JSON only. "
                "If you need a tool, respond with "
                '{"final": false, "assistant_message": "...", "tool_name": "...", "tool_arguments": {...}}. '
                "If you are done, respond with "
                '{"final": true, "assistant_message": "..."}.\n\n'
                "Important rules:\n"
                "- tool_arguments must be a JSON object, not a JSON-encoded string.\n"
                "- Array arguments must be real JSON arrays, not strings.\n"
                "- For a small BEPS test, prefer the provided valid demo payload.\n"
                "- Do not invent tiny 2-point loop axes; BEPS loop fitting needs enough points.\n\n"
                "Valid demo payload for fit_beps_loops_tool:\n"
                f"{json.dumps(demo_beps_payload)}\n\n"
                "Valid demo payload for fit_sho_response_tool:\n"
                f"{json.dumps(demo_sho_payload)}\n\n"
                f"Available MCP tools:\n{json.dumps(tool_catalog, indent=2)}"
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for step in range(1, max_steps + 1):
        response = _timed_call(
            f"ollama step {step}",
            _ollama_chat,
            model=model,
            messages=messages,
            ollama_url=ollama_url,
            timeout=timeout,
            format_json=True,
            stream=ollama_stream,
        )
        content = response["message"]["content"]
        try:
            decision = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama did not return valid JSON:\n{content}") from exc

        assistant_message = decision.get("assistant_message", "").strip()
        if assistant_message:
            print(f"\n[assistant step {step}] {assistant_message}\n")

        if decision.get("final", False):
            if tool_calls_made > 0:
                return assistant_message or "Finished."
            reminder = (
                "You have not called any MCP tool yet. "
                "You must call at least one tool before returning final=true. "
                "If the user asked you to run a BEPS test, use the provided valid demo payload "
                "with fit_beps_loops_tool."
            )
            messages.append({"role": "assistant", "content": json.dumps(decision)})
            messages.append({"role": "user", "content": reminder})
            print("\n[assistant step {}] model tried to finish without a tool call; requesting a real tool run.\n".format(step))
            continue

        tool_name = decision.get("tool_name")
        tool_arguments = _coerce_json_like(decision.get("tool_arguments", {}))
        if not tool_name:
            raise RuntimeError(f"Missing tool_name in model response:\n{json.dumps(decision, indent=2)}")

        tool_result = _timed_call(
            f"tool call {tool_name}",
            client.call_tool,
            tool_name,
            tool_arguments,
        )
        tool_calls_made += 1
        result_text = _extract_tool_text(tool_result)
        result_summary = _summarize_tool_result(tool_result)

        print(f"[tool call] {tool_name}")
        print(json.dumps(_summarize_value(tool_arguments), indent=2))
        print("\n[tool result]")
        print(json.dumps(result_summary, indent=2))
        print()

        messages.append({"role": "assistant", "content": json.dumps(decision)})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Tool result summary:\n"
                    + json.dumps(
                        {
                            "tool_name": tool_name,
                            "result_summary": result_summary,
                        }
                    )
                ),
            }
        )

    raise RuntimeError(f"Reached max steps ({max_steps}) before the model finished.")


def cmd_list_tools(args: argparse.Namespace) -> int:
    with StdioMcpClient(args.server_command, cwd=Path(args.cwd), debug=args.debug) as client:
        tools = client.list_tools()
        print(_pretty_tool_summary(tools))
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    arguments = json.loads(args.args_json)
    with StdioMcpClient(args.server_command, cwd=Path(args.cwd), debug=args.debug) as client:
        result = client.call_tool(args.tool, arguments)
        print(_extract_tool_text(result))
    return 0


def cmd_demo_beps(args: argparse.Namespace) -> int:
    payload = build_demo_beps_payload()
    with StdioMcpClient(args.server_command, cwd=Path(args.cwd), debug=args.debug) as client:
        result = client.call_tool("fit_beps_loops_tool", payload)
        print(_extract_tool_text(result))
    return 0


def cmd_demo_sho(args: argparse.Namespace) -> int:
    payload = build_demo_sho_payload()
    with StdioMcpClient(args.server_command, cwd=Path(args.cwd), debug=args.debug) as client:
        result = client.call_tool("fit_sho_response_tool", payload)
        print(_extract_tool_text(result))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    with StdioMcpClient(args.server_command, cwd=Path(args.cwd), debug=args.debug) as client:
        final_text = run_ollama_tool_loop(
            client=client,
            model=args.model,
            prompt=args.prompt,
            ollama_url=args.ollama_url,
            timeout=args.timeout,
            max_steps=args.max_steps,
            ollama_stream=args.ollama_stream,
        )
        print(final_text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-command",
        default=DEFAULT_SERVER_COMMAND,
        help="Command used to launch the MCP server over stdio.",
    )
    parser.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Working directory to use when launching the MCP server.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print simple client-side progress messages to stderr.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tools", help="List MCP tools exposed by the server.")
    list_parser.set_defaults(func=cmd_list_tools)

    call_parser = subparsers.add_parser("call", help="Call one MCP tool directly with JSON arguments.")
    call_parser.add_argument("--tool", required=True, help="Tool name, for example fit_beps_loops_tool.")
    call_parser.add_argument("--args-json", required=True, help="JSON object passed as tool arguments.")
    call_parser.set_defaults(func=cmd_call)

    demo_beps_parser = subparsers.add_parser("demo-beps", help="Run a synthetic BEPS fit through MCP.")
    demo_beps_parser.set_defaults(func=cmd_demo_beps)

    demo_sho_parser = subparsers.add_parser("demo-sho", help="Run a synthetic SHO fit through MCP.")
    demo_sho_parser.set_defaults(func=cmd_demo_sho)

    chat_parser = subparsers.add_parser("chat", help="Use Ollama to drive an MCP tool loop.")
    chat_parser.add_argument("--model", required=True, help="Ollama model name, for example llama3.1.")
    chat_parser.add_argument("--prompt", required=True, help="Prompt given to the model.")
    chat_parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama /api/chat endpoint.")
    chat_parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds for Ollama.")
    chat_parser.add_argument("--max-steps", type=int, default=6, help="Maximum model/tool turns.")
    chat_parser.add_argument(
        "--ollama-stream",
        action="store_true",
        help="Stream Ollama tokens to stderr while waiting for each response.",
    )
    chat_parser.set_defaults(func=cmd_chat)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
