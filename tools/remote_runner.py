import argparse
import getpass
import os
import shlex
import sys
import tempfile
import time
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as tomllib  # type: ignore

try:
    import paramiko
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "缺少 paramiko 依赖，请先 `uv run pip install paramiko` 或在环境中安装。"
    ) from exc


@dataclass
class RemotePaths:
    workdir: str
    repo: str
    story_path: str
    text_path: str
    output_dir: str
    output_file: str
    subtitle_file: Optional[str]
    log_path: str
    timestamp: str


@dataclass
class LocalAssets:
    config_path: Path
    text_path: Path
    voices: Dict[str, Path]
    output_dir: Path
    temp_files: List[Path]


@dataclass
class RuntimeOverrides:
    device: Optional[str]
    use_fp16: Optional[bool]
    use_cuda_kernel: Optional[bool]
    use_deepspeed: Optional[bool]
    cfg_path: Optional[str]
    model_dir: Optional[str]
    num_workers: Optional[int]
    devices: Optional[str]


SSH_DEFAULT_PORT = 22


def prompt(text: str, default: Optional[str] = None, allow_empty: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{text}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if allow_empty:
            return ""
        print("请输入有效值。\n")


def parse_ssh_command(cmd: str) -> Dict[str, str]:
    tokens = shlex.split(cmd)
    if not tokens:
        raise ValueError("SSH 命令不能为空。")
    if tokens[0] == "ssh":
        tokens = tokens[1:]
    port = SSH_DEFAULT_PORT
    user = None
    host = None
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "-p":
            idx += 1
            if idx >= len(tokens):
                raise ValueError("-p 后缺少端口号。")
            port = int(tokens[idx])
        elif token.startswith("-"):
            raise ValueError(f"暂不支持的 SSH 参数: {token}")
        else:
            host_spec = token
            if "@" in host_spec:
                user, host = host_spec.split("@", 1)
            else:
                host = host_spec
        idx += 1
    if not host:
        raise ValueError("未解析到目标主机，请按照示例输入，如 ssh -p 22774 root@example.com")
    if not user:
        user = getpass.getuser()
    return {"host": host, "user": user, "port": port}


def load_config(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def detect_text_file(source_dir: Path) -> Optional[Path]:
    prefer = source_dir / f"{source_dir.name}.txt"
    if prefer.is_file():
        return prefer
    return None


def detect_voice_files(source_dir: Path) -> Dict[str, Path]:
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    pattern = re.compile(r"^speaker([0-9]+|[A-Za-z]+)$", re.IGNORECASE)
    matches: List[Tuple[Tuple[int, object], str, Path]] = []
    for audio in source_dir.iterdir():
        if audio.suffix.lower() not in audio_exts:
            continue
        m = pattern.match(audio.stem)
        if not m:
            continue
        suffix = m.group(1)
        if suffix.isdigit():
            index = int(suffix)
            voice_name = f"speaker{index}"
            sort_key: Tuple[int, object] = (0, index)
        else:
            voice_name = f"speaker{suffix}"
            sort_key = (1, suffix.lower())
        matches.append((sort_key, voice_name, audio))

    voices = OrderedDict()
    for _, name, audio in sorted(matches, key=lambda item: item[0]):
        voices[name] = audio
    return voices


def build_default_story_config(
    source_dir: Path,
    text_path: Path,
    voices: Dict[str, Path],
) -> Dict[str, object]:
    story_name = source_dir.name
    config: Dict[str, object] = {
        "model": "index-tts-2",
        "ref_audio": "",
        "ref_text": "",
        "gen_text": "",
        "gen_file": text_path.name,
        "remove_silence": True,
        "output_dir": f"/root/autodl-fs/{story_name}",
        "output_file": f"{story_name}.mp3",
        "output_subtitle_file": f"subtitle-{story_name}.json",
        "runtime": {
            "interval_silence": 200,
            "max_text_tokens_per_segment": 120,
            "use_fp16": True,
            "use_cuda_kernel": True,
            "use_deepspeed": False,
        },
        "voices": {},
    }

    in_section = OrderedDict()
    for name, path in voices.items():
        in_section[name] = {"ref_audio": path.name, "ref_text": ""}
    config["voices"] = in_section
    return config


def resolve_voices(
    template: Dict,
    cli_overrides: Sequence[str],
    source_dir: Optional[Path] = None,
    detected: Optional[Dict[str, Path]] = None,
) -> Dict[str, Path]:
    voices = template.get("voices") or {}
    if not voices:
        raise ValueError("配置文件中未定义 voices。")
    overrides: Dict[str, Path] = {}
    for item in cli_overrides:
        if "=" not in item:
            raise ValueError(f"--voice 参数格式应为 name=path，当前为 {item}")
        name, raw_path = item.split("=", 1)
        overrides[name.strip()] = Path(raw_path.strip()).expanduser().resolve()
    result: Dict[str, Path] = {}
    if detected:
        for name, path in detected.items():
            target = overrides.get(name, path)
            target = target.expanduser().resolve()
            if not target.is_file():
                raise FileNotFoundError(f"说话人 {name} 的音频不存在: {target}")
            result[name] = target
            overrides.pop(name, None)
        if overrides:
            extra = ", ".join(overrides.keys())
            raise ValueError(f"--voice 中存在未匹配的说话人: {extra}")
        return result
    audio_exts = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".WAV", ".MP3", ".FLAC", ".M4A", ".OGG")
    for name, params in voices.items():
        default_path = Path(params.get("ref_audio", "")).expanduser()
        default_candidate: Optional[Path] = None
        if default_path and default_path.is_file():
            default_candidate = default_path
        elif Path("autodl-test").joinpath(f"{name}.wav").is_file():
            default_candidate = Path("autodl-test") / f"{name}.wav"
        elif source_dir:
            candidates: List[Path] = []
            remote_hint = params.get("ref_audio")
            if remote_hint:
                candidates.append(source_dir / Path(remote_hint).name)
            for ext in audio_exts:
                candidates.append(source_dir / f"{name}{ext}")
            candidates.extend(source_dir.glob(f"{name}.*"))
            for cand in candidates:
                if cand.is_file():
                    default_candidate = cand
                    break
        if name in overrides:
            selected = overrides[name]
            if not selected.is_file():
                raise FileNotFoundError(f"覆盖的说话人音频不存在: {selected}")
            result[name] = selected
            print(f"使用 CLI 指定的 {name}: {selected}")
            continue
        if default_candidate and default_candidate.is_file():
            result[name] = default_candidate.resolve()
            print(f"自动匹配到说话人 {name} 的音频: {result[name]}")
            continue
        while True:
            raw = prompt(f"说话人 {name} 的本地音频路径", None)
            candidate = Path(raw).expanduser().resolve()
            if candidate.is_file():
                result[name] = candidate
                break
            print(f"未找到文件: {candidate}\n")
    return result


def resolve_text_path(
    template: Dict,
    cli_path: Optional[str],
    source_dir: Optional[Path] = None,
    detected: Optional[Path] = None,
) -> Tuple[Path, bool]:
    if detected:
        resolved = detected.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"文本文件不存在: {resolved}")
        print(f"自动匹配到文本文件: {resolved}")
        return resolved, False
    default_candidates = []
    if cli_path:
        default_candidates.append(cli_path)
    template_file = template.get("gen_file")
    if template_file:
        template_candidate = Path(template_file)
        if not template_candidate.is_absolute() and source_dir:
            default_candidates.append(str((source_dir / template_candidate).resolve()))
        default_candidates.append(template_file)
    if source_dir:
        default_candidates.append(str((source_dir / f"{source_dir.name}.txt").resolve()))
        for txt_file in source_dir.glob("*.txt"):
            default_candidates.append(str(txt_file.resolve()))
    default_candidates.append("autodl-test/bsta.txt")
    default_path = next((p for p in default_candidates if p and Path(p).expanduser().is_file()), None)
    if default_path:
        resolved = Path(default_path).expanduser().resolve()
        print(f"自动匹配到文本文件: {resolved}")
        return resolved, False

    while True:
        raw = prompt("请输入要合成的文本文件路径", None)
        raw_path = Path(raw).expanduser().resolve()
        if raw_path.is_file():
            return raw_path, False
        choice = input("未找到文件，是否直接粘贴文本创建临时文件？(y/n): ").strip().lower()
        if choice == "y":
            print("请输入文本，结束后单独一行输入 END：")
            lines: List[str] = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            tmp_fd, tmp_name = tempfile.mkstemp(prefix="story_", suffix=".txt")
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)
            tmp_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"已写入临时文件: {tmp_path}")
            return tmp_path, True
        else:
            print("请重新输入有效路径。\n")


def build_remote_config(
    template: Dict,
    remote: RemotePaths,
    voices: Dict[str, Path],
    overrides: RuntimeOverrides,
) -> str:
    config_data: Dict[str, object] = {}
    for key, value in template.items():
        if key in {"voices", "runtime"}:
            continue
        config_data[key] = value

    config_data["gen_text"] = ""
    config_data["gen_file"] = remote.text_path
    config_data["output_dir"] = remote.output_dir
    config_data["output_file"] = remote.output_file
    subtitle = remote.subtitle_file or template.get("output_subtitle_file")
    if subtitle:
        config_data["output_subtitle_file"] = subtitle
    config_data["remove_silence"] = bool(template.get("remove_silence", False))

    runtime_cfg = dict(template.get("runtime", {}))
    if overrides.cfg_path:
        runtime_cfg["cfg_path"] = overrides.cfg_path
    elif "cfg_path" not in runtime_cfg:
        runtime_cfg["cfg_path"] = "checkpoints/config.yaml"
    if overrides.model_dir:
        runtime_cfg["model_dir"] = overrides.model_dir
    elif "model_dir" not in runtime_cfg:
        runtime_cfg["model_dir"] = "checkpoints"
    if overrides.device is not None:
        runtime_cfg["device"] = overrides.device
    if overrides.use_fp16 is not None:
        runtime_cfg["use_fp16"] = overrides.use_fp16
    if overrides.use_cuda_kernel is not None:
        runtime_cfg["use_cuda_kernel"] = overrides.use_cuda_kernel
    if overrides.use_deepspeed is not None:
        runtime_cfg["use_deepspeed"] = overrides.use_deepspeed
    if overrides.num_workers is not None:
        runtime_cfg["num_workers"] = overrides.num_workers
    if overrides.devices is not None:
        runtime_cfg["devices"] = overrides.devices

    voices_cfg = {}
    template_voices = template.get("voices", {})
    for name, params in template_voices.items():
        voice_data = dict(params)
        remote_audio = f"{remote.workdir}/{remote.timestamp}-{voices[name].name}"
        voice_data["ref_audio"] = remote_audio
        if voice_data.get("emo_audio"):
            voice_data["emo_audio"] = f"{remote.workdir}/{remote.timestamp}-{Path(voice_data['emo_audio']).name}"
        voices_cfg[name] = voice_data

    data = config_data.copy()
    if runtime_cfg:
        data["runtime"] = runtime_cfg
    data["voices"] = voices_cfg

    return dump_toml(data)


def dump_toml(data: Dict[str, object]) -> str:
    lines: List[str] = []

    def format_value(value):
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
            return f'"{escaped}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return '""'
        if isinstance(value, list):
            return "[" + ", ".join(format_value(v) for v in value) + "]"
        raise TypeError(f"不支持的 TOML 类型: {type(value)!r}")

    def emit_table(table: Dict[str, object], prefix: Optional[str] = None):
        scalar_items = []
        nested_tables = []
        for key, value in table.items():
            if isinstance(value, dict):
                nested_tables.append((key, value))
            else:
                scalar_items.append((key, value))
        for key, value in scalar_items:
            lines.append(f"{key} = {format_value(value)}")
        for key, value in nested_tables:
            lines.append("")
            header = f"[{prefix}.{key}]" if prefix else f"[{key}]"
            lines.append(header)
            emit_table(value, prefix=f"{prefix}.{key}" if prefix else key)

    scalars = {}
    tables = {}
    voice_tables = data.get("voices")
    for key, value in data.items():
        if key == "voices":
            continue
        if isinstance(value, dict):
            tables[key] = value
        else:
            scalars[key] = value

    for key, value in scalars.items():
        lines.append(f"{key} = {format_value(value)}")

    for key, value in tables.items():
        lines.append("")
        lines.append(f"[{key}]")
        emit_table(value)

    if isinstance(voice_tables, dict):
        for voice, params in voice_tables.items():
            lines.append("")
            lines.append(f"[voices.{voice}]")
            emit_table(params)

    return "\n".join(lines) + "\n"


def ensure_remote_dirs(ssh: paramiko.SSHClient, paths: Sequence[str]) -> None:
    for path in paths:
        command = f"mkdir -p '{path}'"
        ssh.exec_command(command)


def upload_files(sftp: paramiko.SFTPClient, files: Dict[str, Path]) -> None:
    for remote_path, local_path in files.items():
        print(f"上传 {local_path} -> {remote_path}")
        sftp.put(str(local_path), remote_path)


def download_file(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path) -> None:
    print(f"下载 {remote_path} -> {local_path}")
    sftp.get(remote_path, str(local_path))


def execute_command(ssh: paramiko.SSHClient, command: str) -> Tuple[int, List[str]]:
    print(f"执行远程命令: {command}")
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    output_lines: List[str] = []
    for line in stdout:
        text = line.rstrip("\n")
        output_lines.append(text)
        print(text)
    err = stderr.read().decode("utf-8")
    if err:
        output_lines.append(err.rstrip("\n"))
        print(err, file=sys.stderr)
    exit_status = stdout.channel.recv_exit_status()
    return exit_status, output_lines


def parse_bool_flag(value: Optional[bool], prompt_text: str) -> Optional[bool]:
    if value is not None:
        return value
    choice = input(f"{prompt_text} [y/n/空=默认]: ").strip().lower()
    if choice == "y":
        return True
    if choice == "n":
        return False
    return None


def prompt_int_choice(prompt_text: str, default: int, minimum: int = 1) -> int:
    if default < minimum:
        default = minimum
    while True:
        raw = input(f"{prompt_text} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("请输入有效的整数。\n")
            continue
        if value < minimum:
            print(f"请输入不小于 {minimum} 的整数。\n")
            continue
        return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="远程语音合成自动化脚本")
    parser.add_argument("--source-dir", help="包含 story/text/语音文件的本地目录")
    parser.add_argument("--config", help="本地 story.toml 模板")
    parser.add_argument("--text-file", dest="text_file", help="默认文本文件路径")
    parser.add_argument("--voice", action="append", default=[], help="覆盖说话人音频，格式 name=path，可重复")
    parser.add_argument("--remote-workdir", help="远程工作目录，存放音频/文本/结果")
    parser.add_argument("--remote-repo", help="远端仓库根目录")
    parser.add_argument("--story-name", default="story", help="生成的配置与文本文件前缀")
    parser.add_argument("--output-basename", help="远端输出音频文件名")
    parser.add_argument("--subtitle-name", help="远端输出字幕文件名")
    parser.add_argument("--local-output", default="outputs/remote", help="本地下载目录")
    parser.add_argument("--device", help="推理设备，如 cuda:0")
    parser.add_argument("--use-fp16", dest="use_fp16", action="store_true")
    parser.add_argument("--no-use-fp16", dest="use_fp16", action="store_false")
    parser.set_defaults(use_fp16=None)
    parser.add_argument("--use-cuda-kernel", dest="use_cuda_kernel", action="store_true")
    parser.add_argument("--no-use-cuda-kernel", dest="use_cuda_kernel", action="store_false")
    parser.set_defaults(use_cuda_kernel=None)
    parser.add_argument("--use-deepspeed", dest="use_deepspeed", action="store_true")
    parser.add_argument("--no-use-deepspeed", dest="use_deepspeed", action="store_false")
    parser.set_defaults(use_deepspeed=None)
    parser.add_argument("--cfg-path", dest="cfg_path")
    parser.add_argument("--model-dir", dest="model_dir")
    parser.add_argument("--num-workers", type=int, help="远端 batch_infer 并行 worker 数量")
    parser.add_argument("--devices", help="远端 batch_infer worker 设备列表，如 cuda:0,cuda:1,cuda:2")
    parser.add_argument("--ssh", help="完整的 SSH 命令或 user@host 格式；可包含 -p 端口")
    parser.add_argument("--password", help="SSH 密码；如留空则按默认方式登录")
    args = parser.parse_args(argv)

    source_dir: Optional[Path] = None
    if args.source_dir:
        source_dir = Path(args.source_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"找不到目录: {source_dir}")
    else:
        default_dir = Path("autodl-test")
        default_dir_str = str(default_dir.resolve()) if default_dir.is_dir() else None
        raw_source = prompt(
            "请输入包含 story/text/语音的本地目录",
            default_dir_str,
            allow_empty=True,
        )
        if raw_source:
            source_dir = Path(raw_source).expanduser().resolve()
            if not source_dir.is_dir():
                raise FileNotFoundError(f"找不到目录: {source_dir}")

    generated_config = False
    detected_text_path: Optional[Path] = None
    detected_voice_map: Optional[Dict[str, Path]] = None
    if source_dir:
        detected_text_path = detect_text_file(source_dir)
        detected_voice_map = detect_voice_files(source_dir)
        if not detected_voice_map:
            raise FileNotFoundError(
                f"在目录 {source_dir} 中未找到命名为 speakerN.* (N 可为数字或字母) 的音频文件。"
            )
        if not detected_text_path:
            raise FileNotFoundError(
                f"在目录 {source_dir} 中未找到 {source_dir.name}.txt 文本文件。"
            )

    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    elif source_dir:
        config_path = source_dir / "story.toml"
    else:
        config_path = Path("autodl-test/story.toml").resolve()

    if source_dir and not config_path.is_file():
        default_config = build_default_story_config(
            source_dir, detected_text_path, detected_voice_map
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(dump_toml(default_config), encoding="utf-8")
        generated_config = True
        print(f"自动生成 story.toml: {config_path}")

    if not config_path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    template = load_config(config_path)

    if source_dir:
        template["gen_file"] = detected_text_path.name
        voice_section = OrderedDict()
        for name, path in detected_voice_map.items():
            voice_section[name] = {"ref_audio": path.name, "ref_text": ""}
        template["voices"] = voice_section

    voices = resolve_voices(
        template,
        args.voice,
        source_dir=source_dir,
        detected=detected_voice_map,
    )
    text_path, text_is_temp = resolve_text_path(
        template,
        args.text_file,
        source_dir=source_dir,
        detected=detected_text_path,
    )

    def derive_remote_workdir() -> str:
        if args.remote_workdir:
            return args.remote_workdir.rstrip("/")
        template_dir = template.get("output_dir")
        if isinstance(template_dir, str) and template_dir.strip():
            if template_dir.startswith("/"):
                return template_dir.rstrip("/")
            base = "/root/autodl-fs"
            return f"{base}/{template_dir.strip('/')}"
        if source_dir:
            return f"/root/autodl-fs/{source_dir.name}"
        return "/root/autodl-fs"

    remote_workdir = derive_remote_workdir()
    print(f"使用远程工作目录: {remote_workdir}")

    def derive_remote_repo() -> str:
        if args.remote_repo:
            return args.remote_repo.rstrip("/")
        cfg_repo = template.get("remote_repo")
        if isinstance(cfg_repo, str) and cfg_repo.strip():
            return cfg_repo.rstrip("/")
        return "/root/index-tts"

    remote_repo = derive_remote_repo()
    print(f"使用远程仓库根目录: {remote_repo}")

    story_name = args.story_name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    default_output = args.output_basename or template.get("output_file", "")
    if not default_output:
        default_output = f"{story_name}-{timestamp}.wav"
    subtitle_name = args.subtitle_name or template.get("output_subtitle_file")
    if subtitle_name and not subtitle_name.strip():
        subtitle_name = None
    if subtitle_name and '{timestamp}' in subtitle_name:
        subtitle_name = subtitle_name.format(timestamp=timestamp, story=story_name)
    elif subtitle_name is None and template.get("output_subtitle_file"):
        subtitle_name = f"subtitle-{story_name}-{timestamp}.json"

    if args.ssh:
        login = parse_ssh_command(args.ssh)
    else:
        ssh_command = prompt("请输入 SSH 连接命令 (例如 ssh -p 22774 root@example.com)")
        login = parse_ssh_command(ssh_command)

    if args.password is not None:
        password = args.password or None
    else:
        password = getpass.getpass(f"输入 {login['user']}@{login['host']} 的密码: ")

    use_fp16 = parse_bool_flag(args.use_fp16, "是否开启 FP16")
    use_cuda_kernel = parse_bool_flag(args.use_cuda_kernel, "是否使用 BigVGAN CUDA kernel")
    use_deepspeed = parse_bool_flag(args.use_deepspeed, "是否启用 DeepSpeed")

    runtime_template = template.get("runtime", {}) if isinstance(template.get("runtime"), dict) else {}
    default_workers_cfg = runtime_template.get("num_workers")
    try:
        default_workers = int(default_workers_cfg)
    except (TypeError, ValueError):
        default_workers = 3
    num_workers = args.num_workers if args.num_workers is not None else prompt_int_choice(
        "并行 worker 数量", default_workers if default_workers > 0 else 3, minimum=1
    )

    overrides = RuntimeOverrides(
        device=args.device,
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        num_workers=num_workers,
        devices=args.devices,
    )

    remote_paths = RemotePaths(
        workdir=remote_workdir,
        repo=remote_repo,
        story_path=f"{remote_workdir}/{story_name}-{timestamp}.toml",
        text_path=f"{remote_workdir}/{story_name}-{timestamp}.txt",
        output_dir=remote_workdir,
        output_file=default_output,
        subtitle_file=subtitle_name,
        log_path=f"{remote_workdir}/{story_name}-{timestamp}.log",
        timestamp=timestamp,
    )

    config_content = build_remote_config(template, remote_paths, voices, overrides)

    local_output_dir = Path(args.local_output).expanduser()
    local_output_dir.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(prefix="story_remote_", suffix=".toml")
    os.close(tmp_fd)
    local_config_tmp = Path(tmp_name)
    local_config_tmp.write_text(config_content, encoding="utf-8")
    print(f"已生成临时配置: {local_config_tmp}")

    temp_files = [local_config_tmp]
    if text_is_temp:
        temp_files.append(text_path)

    assets = LocalAssets(
        config_path=local_config_tmp,
        text_path=text_path,
        voices=voices,
        output_dir=local_output_dir,
        temp_files=temp_files,
    )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("连接远程服务器...")
    client.connect(
        hostname=login["host"],
        port=login["port"],
        username=login["user"],
        password=password,
        look_for_keys=not bool(password),
    )
    try:
        sftp = client.open_sftp()
        ensure_remote_dirs(client, [remote_paths.workdir, remote_paths.repo])
        upload_map = {
            remote_paths.story_path: assets.config_path,
            remote_paths.text_path: assets.text_path,
        }
        for voice_name, local_voice in voices.items():
            remote_voice_path = f"{remote_paths.workdir}/{remote_paths.timestamp}-{local_voice.name}"
            upload_map[remote_voice_path] = local_voice
        upload_files(sftp, upload_map)

        env_prefix = "export PATH=\"/root/miniconda3/bin:$PATH\""
        if template.get("hf_endpoint"):
            env_prefix += f" && export HF_ENDPOINT=\"{template['hf_endpoint']}\""
        command = (
            f"cd '{remote_paths.repo}' && {env_prefix} && "
            f"(PYTHONPATH='$PWD' uv run python tools/batch_infer.py --config '{remote_paths.story_path}' "
            f"2>&1 | tee '{remote_paths.log_path}')"
        )
        start_ts = time.time()
        exit_status, output_lines = execute_command(client, command)
        wall_elapsed = time.time() - start_ts

        log_dest = (
            source_dir / f"{story_name}-{timestamp}.log"
            if source_dir
            else local_output_dir / f"{story_name}-{timestamp}.log"
        )
        log_dest.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        print(f"本地日志已保存: {log_dest}")

        def find_metric(keyword: str) -> Optional[str]:
            for line in reversed(output_lines):
                if keyword in line:
                    return line.strip()
            return None

        length_line = find_metric("Generated audio length") or find_metric("总时长")
        total_time_line = find_metric("Total inference time")
        rtf_line = find_metric("RTF")

        generated_duration = None
        if length_line:
            match = re.search(r"([0-9.]+)\s*秒", length_line)
            if match:
                generated_duration = float(match.group(1))

        model_elapsed = None
        if total_time_line:
            match = re.search(r"([0-9.]+)\s*seconds", total_time_line)
            if match:
                model_elapsed = float(match.group(1))

        reported_rtf = None
        if rtf_line:
            match = re.search(r"RTF:\s*([0-9.]+)", rtf_line)
            if match:
                reported_rtf = float(match.group(1))

        for label, value in [
            ("生成时长", length_line),
            ("总推理耗时", total_time_line),
            ("RTF", rtf_line),
        ]:
            if value:
                print(f"{label}: {value}")

        print(f"任务总耗时: {wall_elapsed:.2f} 秒")
        overall_rtf = None
        if generated_duration and generated_duration > 0:
            overall_rtf = wall_elapsed / generated_duration
            print(f"任务总体 RTF: {overall_rtf:.4f}")

        summary_dest = (
            source_dir / f"{story_name}-{timestamp}-summary.txt"
            if source_dir
            else local_output_dir / f"{story_name}-{timestamp}-summary.txt"
        )
        with summary_dest.open("w", encoding="utf-8") as fh:
            if generated_duration is not None:
                fh.write(f"Generated audio length: {generated_duration:.2f} seconds\n")
            if model_elapsed is not None:
                fh.write(f"Model total inference time: {model_elapsed:.2f} seconds\n")
            fh.write(f"Wall clock elapsed: {wall_elapsed:.2f} seconds\n")
            if reported_rtf is not None:
                fh.write(f"Reported RTF: {reported_rtf:.4f}\n")
            if overall_rtf is not None:
                fh.write(f"Overall RTF: {overall_rtf:.4f}\n")
        print(f"汇总信息已保存: {summary_dest}")

        if exit_status != 0:
            raise SystemExit(f"远程命令执行失败，退出码 {exit_status}")

        files_to_fetch: Dict[str, Path] = {
            remote_paths.log_path: source_dir / Path(remote_paths.log_path).name if source_dir else Path(remote_paths.log_path).name,
            f"{remote_paths.output_dir}/{remote_paths.output_file}": source_dir / Path(remote_paths.output_file).name if source_dir else Path(remote_paths.output_file).name,
        }

        if remote_paths.subtitle_file:
            if remote_paths.subtitle_file.startswith("/"):
                remote_sub_path = remote_paths.subtitle_file
            else:
                remote_sub_path = f"{remote_paths.output_dir}/{remote_paths.subtitle_file}"
            files_to_fetch[remote_sub_path] = (
                source_dir / Path(remote_paths.subtitle_file).name
                if source_dir
                else Path(remote_paths.subtitle_file).name
            )

        for remote_path, local_path in files_to_fetch.items():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            download_file(sftp, remote_path, local_path)
        print("全部完成。")
    finally:
        client.close()
        for tmp in assets.temp_files:
            if tmp.exists():
                tmp.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
