"""OpenAI-compatible CLI translator for podcast transcripts.

This script reads an input English transcript file, injects it into a
localization + TTS-optimization prompt (Chinese), and calls an
OpenAI-compatible Chat Completions endpoint to produce the final output.

Features:
- OpenAI-compatible: works with OpenAI or third-party compatible servers
- Config via env-file: base URL, API key, default model
- Custom prompt file supported (defaults to the built-in prompt if omitted)

Quick usage:
  # 1) Edit tools/openai_translator.env with your base URL / API key
  # 2) Run translation
  .venv/bin/python tools/translate_openai_cli.py \
      --env tools/openai_translator.env \
      --input /path/to/english.txt \
      --output /path/to/translation_zh.md

With a custom model or prompt file:
  .venv/bin/python tools/translate_openai_cli.py \
      --env tools/openai_translator.env \
      --input drugs-raw.txt \
      --output translation_zh.md \
      --model gpt-4o-mini \
      --prompt-file tools/prompts/zh_localize_tts_prompt.md

Notes:
- The prompt file should include the token {text} where the full English
  transcript will be injected. The default prompt file shipped with this repo
  already follows this convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional
from urllib import request, error

# Optional JSON repair dependency
try:  # pragma: no cover - optional runtime dep
    from json_repair import repair_json as _repair_json
except Exception:  # noqa: BLE001
    _repair_json = None  # type: ignore


DEFAULT_ENV_PATH = Path("tools/openai_translator.env")
DEFAULT_PROMPT_PATH = Path("tools/prompts/zh_localize_tts_prompt.md")


def load_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.is_file():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible CLI translator (Chinese localization + TTS optimization)")
    p.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH, help="Env file with OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL")
    p.add_argument("--input", type=Path, required=True, help="Path to the English transcript (text file)")
    p.add_argument("--output", type=Path, required=True, help="Where to write the Markdown translation")
    p.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_PATH, help="Prompt file containing {text} placeholder")
    p.add_argument("--base-url", help="Override base URL (e.g., https://api.openai.com)")
    p.add_argument("--api-key", help="Override API key")
    p.add_argument("--model", help="Override model name")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    p.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=65000,
        help="Max tokens for response (default: 65000)",
    )
    p.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds")
    # JSON handling
    p.add_argument("--emit-json", action="store_true", help="Expect and save JSON output based on the prompt's schema")
    p.add_argument("--json-output", type=Path, help="Where to write the raw JSON response (if --emit-json)")
    p.add_argument("--script-out", type=Path, help="Where to write the extracted TTS script text (if --emit-json)")
    p.add_argument("--analysis-out", type=Path, help="Where to write the extracted analysis text (if --emit-json)")
    # Speaker tag handling for TTS handoff
    p.add_argument(
        "--rewrite-speaker",
        action="append",
        default=[],
        help="Rewrite speaker tags in final script, format old=new; can repeat",
    )
    p.add_argument(
        "--ensure-speakers-from",
        type=Path,
        help="Directory containing speaker*.{wav,mp3,flac,m4a,ogg,aac}; ensures all tags used in script exist",
    )
    p.add_argument(
        "--fallback-narrator",
        help="If [speakerY] appears but is unavailable, rewrite it to this tag (e.g., speaker0)",
    )
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> Dict[str, str]:
    env = load_env_file(Path(args.env)) if args.env else {}
    cfg = {
        "base_url": args.base_url or env.get("OPENAI_BASE_URL") or "https://api.openai.com",
        "api_key": args.api_key or env.get("OPENAI_API_KEY") or "",
        "model": args.model or env.get("OPENAI_MODEL") or "gpt-4o-mini",
    }
    return cfg


def build_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    # Accept either base ending with /v1 or without; ensure final path includes /v1/chat/completions
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def post_chat_completions(
    endpoint: str,
    api_key: str,
    payload: Dict,
    *,
    timeout: int = 300,
) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8", errors="replace"))
    except error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTPError {e.code}: {text}") from e
    except error.URLError as e:
        raise SystemExit(f"URLError: {e}") from e


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"输入文件不存在: {args.input}")
    if not args.prompt_file.is_file():
        raise SystemExit(f"提示词文件不存在: {args.prompt_file}")

    cfg = resolve_config(args)
    if not cfg["api_key"]:
        print("[warn] OPENAI_API_KEY 未配置（env 或 --api-key）。若目标服务需要鉴权，请设置。")

    endpoint = build_endpoint(cfg["base_url"])

    english_text = args.input.read_text(encoding="utf-8")
    prompt_template = args.prompt_file.read_text(encoding="utf-8")

    # Replace placeholder {text} safely without format()-style expansion
    if "{text}" not in prompt_template:
        print("[warn] 提示词中未发现 {text} 占位符，将把原文附加在提示词末尾。")
        full_user = f"{prompt_template}\n\n{text}\n{english_text}"
    else:
        full_user = prompt_template.replace("{text}", english_text)

    # You may choose to add a minimal system message.
    system_msg = (
        "You are a senior cross-cultural localization expert and a world-class podcast producer."
    )

    payload: Dict[str, Optional[object]] = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_user},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = int(args.max_tokens)

    print(f"调用: {endpoint} (model={cfg['model']})", file=sys.stderr)
    data = post_chat_completions(endpoint, cfg["api_key"], payload, timeout=args.timeout)

    if not isinstance(data, dict) or "choices" not in data:
        out_path = args.output if isinstance(args.output, Path) else Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("API 返回异常，已写入原始响应到输出文件以便检查。")

    content = ""
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except Exception:
        content = ""

    if not content:
        out_path = args.output if isinstance(args.output, Path) else Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit("返回中未找到内容，已写入原始响应 JSON。")

    # Decide output mode
    if not args.emit_json:
        out_path = args.output if isinstance(args.output, Path) else Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"已写出: {out_path} ({len(content)} chars)")
        return

    # Emit JSON and split to text files
    text = content.strip()
    # Strip potential code-fence if model still returns with ```json ... ```
    if text.startswith("```"):
        # remove first fence line and trailing fence
        lines = text.splitlines()
        # drop leading fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # drop trailing fence if exists
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n\n".join(lines).strip()

    # Attempt to load JSON; use json-repair if available, then fallback to braces slice
    def _try_parse_json(payload: str) -> Optional[Dict]:
        try:
            return json.loads(payload)
        except Exception:
            return None

    obj = _try_parse_json(text)
    if obj is None and _repair_json is not None:
        try:
            repaired = _repair_json(text)
            obj = _try_parse_json(repaired)
        except Exception:
            obj = None
    if obj is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            slice_text = text[start : end + 1]
            obj = _try_parse_json(slice_text)
            if obj is None and _repair_json is not None:
                try:
                    obj = _try_parse_json(_repair_json(slice_text))
                except Exception:
                    obj = None
    if obj is None:
        out_path = args.output if isinstance(args.output, Path) else Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        hint = "（建议安装 json-repair：pip install json-repair）"
        raise SystemExit(f"无法解析为 JSON，已写入原始文本供检查。{hint}")

    # Paths
    base_output = args.output if isinstance(args.output, Path) else Path(args.output)
    json_path = args.json_output or (base_output if base_output.suffix == ".json" else base_output.with_suffix(".json"))
    script_path = args.script_out or base_output.with_name(base_output.stem + "_script.txt")
    analysis_path = args.analysis_out or base_output.with_name(base_output.stem + "_analysis.txt")

    # Write JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # Extract fields
    script = ""
    analysis = ""
    if isinstance(obj, dict):
        script = (
            obj.get("script")
            or obj.get("final_script")
            or obj.get("tts_script")
            or ""
        )
        a = obj.get("analysis")
        if isinstance(a, dict):
            sg = a.get("style_guide") or ""
            sc = a.get("self_critique") or ""
            kd = a.get("key_decisions") or ""
            parts = []
            if sg:
                parts.append("# 项目风格指南\n\n" + sg)
            if sc:
                parts.append("# 批判性反思报告\n\n" + sc)
            if kd:
                parts.append("# 关键决策亮点\n\n" + kd)
            analysis = "\n\n---\n\n".join(parts)

    # Optionally rewrite speaker tags in script to match available samples
    def _collect_available_speakers(dir_path: Path):
        tags = set()
        if not dir_path or not dir_path.is_dir():
            return tags
        import re
        for entry in dir_path.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}:
                continue
            m = re.match(r"^(speaker[0-9A-Za-z]+)$", entry.stem)
            if m:
                tags.add(m.group(1))
        return tags

    def _rewrite_script_tags(text: str, mapping: Dict[str, str]) -> str:
        import re
        pat = re.compile(r"^(\s*)\[(?P<tag>[^\]]+)\](?P<rest>.*)$")
        lines = []
        for raw in text.splitlines():
            m = pat.match(raw)
            if not m:
                lines.append(raw)
                continue
            old = m.group("tag").strip()
            new = mapping.get(old, old)
            lines.append(f"{m.group(1)}[{new}]{m.group('rest')}")
        return "\n".join(lines)

    def _used_tags(text: str) -> set:
        import re
        tags = set()
        for line in text.splitlines():
            m = re.match(r"^\s*\[([^\]]+)\]", line)
            if m:
                tags.add(m.group(1).strip())
        return tags

    if script:
        rewrite_map: Dict[str, str] = {}
        # CLI remaps first
        for item in args.rewrite_speaker or []:
            if "=" not in item:
                print(f"[warn] 忽略无效 --rewrite-speaker: {item}")
                continue
            old, new = item.split("=", 1)
            rewrite_map[old.strip()] = new.strip()
        # narrator fallback if requested
        if args.fallback_narrator:
            used = _used_tags(script)
            avail = _collect_available_speakers(args.ensure_speakers_from) if args.ensure_speakers_from else set()
            if "speakerY" in used and (not avail or "speakerY" not in avail):
                # only apply if fallback exists in available set (if provided)
                target = args.fallback_narrator.strip()
                if not avail or target in avail:
                    rewrite_map.setdefault("speakerY", target)
        # Apply rewrite
        if rewrite_map:
            script = _rewrite_script_tags(script, rewrite_map)
        # Strict check against available samples
        if args.ensure_speakers_from:
            avail = _collect_available_speakers(args.ensure_speakers_from)
            missing = sorted(t for t in _used_tags(script) if t not in avail)
            if missing:
                hint = ", ".join(missing)
                raise SystemExit(f"脚本中存在未提供样本的说话人标签: {hint}")
        script_path.write_text(script, encoding="utf-8")
    if analysis:
        analysis_path.write_text(analysis, encoding="utf-8")

    print(f"已写出 JSON: {json_path}")
    if script:
        print(f"已写出脚本 TXT: {script_path}")
    if analysis:
        print(f"已写出分析 TXT: {analysis_path}")


if __name__ == "__main__":
    main()
