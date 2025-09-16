import argparse
import getpass
import os
import shlex
import sys
import tempfile
import time
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


def resolve_voices(template: Dict, cli_overrides: Sequence[str]) -> Dict[str, Path]:
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
    for name, params in voices.items():
        default_path = Path(params.get("ref_audio", "")).expanduser()
        default_candidate: Optional[Path] = None
        if default_path and default_path.is_file():
            default_candidate = default_path
        elif Path("autodl-test").joinpath(f"{name}.wav").is_file():
            default_candidate = Path("autodl-test") / f"{name}.wav"
        prompt_default = overrides.get(name, default_candidate)
        prompt_default_str = str(prompt_default) if prompt_default else ""
        while True:
            raw = prompt(f"说话人 {name} 的本地音频路径", prompt_default_str or None)
            candidate = Path(raw).expanduser().resolve()
            if candidate.is_file():
                result[name] = candidate
                break
            print(f"未找到文件: {candidate}\n")
    return result


def resolve_text_path(template: Dict, cli_path: Optional[str]) -> Tuple[Path, bool]:
    default_candidates = [cli_path]
    template_file = template.get("gen_file")
    if template_file:
        default_candidates.append(template_file)
    default_candidates.append("autodl-test/bsta.txt")
    default_path = next((p for p in default_candidates if p and Path(p).expanduser().is_file()), None)

    while True:
        raw = prompt("请输入要合成的文本文件路径", default_path)
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


def execute_command(ssh: paramiko.SSHClient, command: str) -> int:
    print(f"执行远程命令: {command}")
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    for line in stdout:
        print(line.rstrip())
    err = stderr.read().decode("utf-8")
    if err:
        print(err, file=sys.stderr)
    return stdout.channel.recv_exit_status()


def parse_bool_flag(value: Optional[bool], prompt_text: str) -> Optional[bool]:
    if value is not None:
        return value
    choice = input(f"{prompt_text} [y/n/空=默认]: ").strip().lower()
    if choice == "y":
        return True
    if choice == "n":
        return False
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="远程语音合成自动化脚本")
    parser.add_argument("--config", default="autodl-test/story.toml", help="本地 story.toml 模板")
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
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    template = load_config(config_path)

    voices = resolve_voices(template, args.voice)
    text_path, text_is_temp = resolve_text_path(template, args.text_file)

    remote_workdir_default = args.remote_workdir or template.get("output_dir") or "/root/autodl-fs"
    remote_workdir = prompt("远程工作目录", remote_workdir_default).rstrip("/")
    remote_repo = args.remote_repo or prompt("远程仓库根目录", "~/Development/index-tts")
    remote_repo = os.path.expanduser(remote_repo.rstrip("/"))

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

    ssh_command = prompt("请输入 SSH 连接命令 (例如 ssh -p 22774 root@example.com)")
    login = parse_ssh_command(ssh_command)
    password = getpass.getpass(f"输入 {login['user']}@{login['host']} 的密码: ")

    use_fp16 = parse_bool_flag(args.use_fp16, "是否开启 FP16")
    use_cuda_kernel = parse_bool_flag(args.use_cuda_kernel, "是否使用 BigVGAN CUDA kernel")
    use_deepspeed = parse_bool_flag(args.use_deepspeed, "是否启用 DeepSpeed")

    overrides = RuntimeOverrides(
        device=args.device,
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
    )

    remote_paths = RemotePaths(
        workdir=remote_workdir,
        repo=remote_repo,
        story_path=f"{remote_workdir}/{story_name}-{timestamp}.toml",
        text_path=f"{remote_workdir}/{story_name}-{timestamp}.txt",
        output_dir=remote_workdir,
        output_file=default_output,
        subtitle_file=subtitle_name,
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
        look_for_keys=False,
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

        command = (
            f"cd '{remote_paths.repo}' && PYTHONPATH='$PWD' "
            f"uv run python tools/batch_infer.py --config '{remote_paths.story_path}'"
        )
        exit_status = execute_command(client, command)
        if exit_status != 0:
            raise SystemExit(f"远程命令执行失败，退出码 {exit_status}")

        remote_audio_path = f"{remote_paths.output_dir}/{remote_paths.output_file}"
        local_audio_path = local_output_dir / Path(remote_paths.output_file).name
        download_file(sftp, remote_audio_path, local_audio_path)
        if remote_paths.subtitle_file:
            if remote_paths.subtitle_file.startswith("/"):
                remote_sub_path = remote_paths.subtitle_file
            else:
                remote_sub_path = f"{remote_paths.output_dir}/{remote_paths.subtitle_file}"
            local_sub_path = local_output_dir / Path(remote_paths.subtitle_file).name
            download_file(sftp, remote_sub_path, local_sub_path)
        print("全部完成。")
    finally:
        client.close()
        for tmp in assets.temp_files:
            if tmp.exists():
                tmp.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
