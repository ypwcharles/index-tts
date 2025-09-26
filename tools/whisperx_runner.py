import argparse
import json
import getpass
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from argparse import BooleanOptionalAction

import paramiko
from paramiko.client import SSHClient
from paramiko.sftp_client import SFTPClient
from stat import S_ISDIR

DEFAULT_ENV_PATH = Path(__file__).resolve().parent / "whisperx.env"
DEFAULT_LOCAL_OUTPUT = "outputs/remote"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload audio, run WhisperX diarized transcription remotely, and download outputs.",
    )
    parser.add_argument("audio", nargs="+", help="Local audio file(s) for transcription.")
    parser.add_argument(
        "--ssh",
        dest="ssh",
        help="SSH command or host spec (e.g. 'ssh -p 22774 root@example.com' or 'root@example.com').",
    )
    parser.add_argument(
        "--password",
        dest="password",
        help="Password for SSH login; prompt if omitted.",
    )
    parser.add_argument(
        "--remote-dir",
        dest="remote_dir",
        default="/root/autodl-fs/whisperX",
        help="Remote base directory to store uploads and outputs.",
    )
    parser.add_argument(
        "--remote-project",
        dest="remote_project",
        default="/root/autodl-fs/whisperX",
        help="Remote WhisperX project directory with uv environment.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="large-v3",
        help="WhisperX model name.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
        help="Batch size for WhisperX inference.",
    )
    parser.add_argument(
        "--compute-type",
        dest="compute_type",
        default="float32",
        choices=["float16", "float32", "int8"],
        help="Compute type used by WhisperX.",
    )
    parser.add_argument(
        "--language",
        dest="language",
        help="Force language code for transcription (optional).",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        help="Hugging Face token for diarization models (optional).",
    )
    parser.add_argument(
        "--min-speakers",
        dest="min_speakers",
        type=int,
        help="Minimum number of speakers for diarization.",
    )
    parser.add_argument(
        "--max-speakers",
        dest="max_speakers",
        type=int,
        help="Maximum number of speakers for diarization.",
    )
    parser.add_argument(
        "--diarize-model",
        dest="diarize_model",
        default="pyannote/speaker-diarization-3.1",
        help="Speaker diarization model identifier.",
    )
    parser.add_argument(
        "--no-diarize",
        dest="no_diarize",
        action="store_true",
        help="Disable speaker diarization.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="cuda",
        help="Device passed to WhisperX (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--local-output",
        dest="local_output",
        default=DEFAULT_LOCAL_OUTPUT,
        help="Local directory to store downloaded transcripts and logs.",
    )
    parser.add_argument(
        "--keep-remote",
        dest="keep_remote",
        action="store_true",
        help="Keep remote run directory after download.",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=0,
        help="Optional timeout (seconds) for the remote command; 0 means wait indefinitely.",
    )
    parser.add_argument(
        "--env-file",
        dest="env_file",
        default=str(DEFAULT_ENV_PATH),
        help="Environment file containing credentials and defaults.",
    )
    parser.add_argument(
        "--upgrade-torch",
        dest="upgrade_torch",
        action=BooleanOptionalAction,
        default=False,
        help="Ensure torch/torchvision/torchaudio match the requested versions (default: off).",
    )
    parser.add_argument(
        "--torch-version",
        dest="torch_version",
        default="2.8.0",
        help="Target torch version when --upgrade-torch is enabled.",
    )
    parser.add_argument(
        "--torchvision-version",
        dest="torchvision_version",
        default="0.19.0",
        help="Target torchvision version when --upgrade-torch is enabled.",
    )
    parser.add_argument(
        "--torchaudio-version",
        dest="torchaudio_version",
        default="2.8.0",
        help="Target torchaudio version when --upgrade-torch is enabled.",
    )
    parser.add_argument(
        "--torch-index-url",
        dest="torch_index_url",
        default=None,
        help="Override the pip index url used for torch packages (defaults to PyTorch CUDA 12.8 wheels).",
    )
    parser.add_argument(
        "--cudnn-root",
        dest="cudnn_root",
        default=None,
        help="Optional cuDNN root directory whose lib folder will be prepended to LD_LIBRARY_PATH.",
    )
    return parser.parse_args()


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


def env_bool(env: Dict[str, str], key: str) -> Optional[bool]:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def parse_ssh_command(cmd: str) -> Dict[str, str]:
    tokens = shlex.split(cmd)
    if not tokens:
        raise ValueError("SSH 命令不能为空。")
    if tokens[0] == "ssh":
        tokens = tokens[1:]
    port = 22
    user: Optional[str] = None
    host: Optional[str] = None
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
            spec = token
            if "@" in spec:
                user, host = spec.split("@", 1)
            else:
                host = spec
        idx += 1
    if not host:
        raise ValueError("未解析到目标主机，请输入类似 root@example.com 的格式。")
    if not user:
        user = getpass.getuser()
    return {"host": host, "user": user, "port": str(port)}


def ensure_remote_dirs(ssh: SSHClient, paths: Sequence[str]) -> None:
    for path in paths:
        command = f"mkdir -p {shlex.quote(path)}"
        ssh.exec_command(command)


def upload_files(sftp: SFTPClient, mapping: Dict[str, Path]) -> None:
    for remote_path, local_path in mapping.items():
        print(f"上传 {local_path} -> {remote_path}")
        sftp.put(str(local_path), remote_path)


def execute_command(
    ssh: SSHClient,
    command: str,
    timeout: Optional[int] = None,
) -> List[str]:
    # Stream both stdout and stderr in real time to avoid the appearance of "hangs"
    # when libraries (e.g., huggingface_hub/transformers) print progress to stderr.
    start_ts = time.time()
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    chan = stdout.channel
    lines: List[str] = []

    def _drain(buf: bytes, is_err: bool = False) -> None:
        if not buf:
            return
        text = buf.decode("utf-8", errors="replace")
        for line in text.splitlines():
            lines.append(line)
            if is_err:
                print(line, file=sys.stderr)
            else:
                print(line)

    while True:
        if timeout and (time.time() - start_ts) > timeout:
            try:
                chan.close()
            finally:
                raise TimeoutError("远程命令执行超时")
        # Read any available data from stdout/stderr
        if chan.recv_ready():
            _drain(chan.recv(4096), is_err=False)
        if chan.recv_stderr_ready():
            _drain(chan.recv_stderr(4096), is_err=True)
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        time.sleep(0.05)

    # Ensure any trailing buffers are consumed
    while chan.recv_ready():
        _drain(chan.recv(4096), is_err=False)
    while chan.recv_stderr_ready():
        _drain(chan.recv_stderr(4096), is_err=True)

    exit_status = chan.recv_exit_status()
    if exit_status != 0:
        raise RuntimeError(f"远程命令返回非零状态: {exit_status}")
    return lines


def download_outputs(
    sftp: SFTPClient,
    remote_dir: str,
    local_dir: Path,
    skip: Sequence[str],
) -> List[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for entry in sftp.listdir_attr(remote_dir):
        if S_ISDIR(entry.st_mode):
            continue
        if entry.filename in skip:
            continue
        remote_path = f"{remote_dir}/{entry.filename}"
        local_path = local_dir / entry.filename
        print(f"下载 {remote_path} -> {local_path}")
        sftp.get(remote_path, str(local_path))
        downloaded.append(local_path)
    return downloaded


def build_torch_upgrade_command(
    remote_project: str,
    torch_version: Optional[str],
    torchvision_version: Optional[str],
    torchaudio_version: Optional[str],
    index_url: Optional[str],
) -> Optional[str]:
    packages: List[str] = []
    if torch_version:
        packages.append(f"torch=={torch_version}")
    if torchvision_version:
        packages.append(f"torchvision=={torchvision_version}")
    if torchaudio_version:
        packages.append(f"torchaudio=={torchaudio_version}")
    if not packages:
        return None
    cmd: List[str] = ["uv", "pip", "install", "--upgrade"]
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(packages)
    quoted = " ".join(shlex.quote(part) for part in cmd)
    segments = [f"cd {shlex.quote(remote_project)}", quoted]
    full = " && ".join(segments)
    return f"bash -lc {shlex.quote(full)}"


def _is_diarization_json(path: Path) -> bool:
    """Heuristically detect a WhisperX diarization JSON (segments -> words with speaker)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segs = data.get("segments")
        if not isinstance(segs, list):
            return False
        for seg in segs:
            words = seg.get("words") if isinstance(seg, dict) else None
            if not isinstance(words, list):
                continue
            for w in words:
                if isinstance(w, dict) and "speaker" in w and w.get("end") is not None:
                    return True
        return False
    except Exception:
        return False


def _collect_speakers_from_json(path: Path) -> Dict[str, str]:
    """Return mapping of raw speaker labels -> normalized tags (e.g., SPEAKER_00 -> speaker0)."""
    result: Dict[str, str] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        speakers = []
        for seg in data.get("segments", []):
            for w in seg.get("words", []) or []:
                s = w.get("speaker")
                if isinstance(s, str):
                    speakers.append(s)
        seen = []
        for s in speakers:
            if s in seen:
                continue
            seen.append(s)
        def _norm(label: str) -> str:
            # Try to extract trailing number
            import re as _re
            m = _re.search(r"(\d+)$", label)
            if m:
                return f"speaker{int(m.group(1))}"
            return f"speaker{len(result)}" if label not in result else result[label]
        for s in seen:
            result[s] = _norm(s)
    except Exception:
        pass
    return result


def build_whisperx_command(
    remote_project: str,
    remote_audio_paths: Sequence[str],
    model: str,
    batch_size: int,
    compute_type: str,
    output_dir: str,
    device: str,
    hf_token: Optional[str] = None,
    language: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarize_model: Optional[str] = None,
    diarize: bool = True,
    hf_endpoint: Optional[str] = None,
    cudnn_root: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
) -> str:
    cmd: List[str] = ["uv", "run", "python", "-m", "whisperx"]
    cmd.extend(remote_audio_paths)
    cmd.extend(["--model", model, "--batch_size", str(batch_size), "--compute_type", compute_type])
    cmd.extend(["--output_dir", output_dir, "--device", device, "--verbose", "True"])
    if diarize:
        cmd.append("--diarize")
        if diarize_model:
            cmd.extend(["--diarize_model", diarize_model])
        if min_speakers is not None:
            cmd.extend(["--min_speakers", str(min_speakers)])
        if max_speakers is not None:
            cmd.extend(["--max_speakers", str(max_speakers)])
    if language:
        cmd.extend(["--language", language])
    if hf_token:
        cmd.extend(["--hf_token", hf_token])
    cmd.extend(["--print_progress", "True"])
    quoted = " ".join(shlex.quote(part) for part in cmd)
    prefix: List[str] = []
    # Export selected environment variables to influence cache locations, endpoints, etc.
    if env_vars:
        for k, v in env_vars.items():
            if v:
                prefix.append(f"export {k}={shlex.quote(v)}")
    if hf_endpoint:
        prefix.append(f"export HF_ENDPOINT={shlex.quote(hf_endpoint)}")
    if cudnn_root:
        prefix.append(f"export CUDNN_ROOT={shlex.quote(cudnn_root)}")
        prefix.append('export LD_LIBRARY_PATH=${CUDNN_ROOT}/lib:${LD_LIBRARY_PATH}')
        prefix.append('export LIBRARY_PATH=${CUDNN_ROOT}/lib:${LIBRARY_PATH}')
    prefix.append(quoted)
    full_cmd = " && ".join(prefix)
    segments = [f"cd {shlex.quote(remote_project)}", full_cmd]
    full = " && ".join(segments)
    return f"bash -lc {shlex.quote(full)}"


def main() -> None:
    args = parse_args()

    local_audio_paths: List[Path] = []
    for item in args.audio:
        path = Path(item).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"未找到音频文件: {path}")
        local_audio_paths.append(path)

    if len(local_audio_paths) != 1:
        raise ValueError("当前脚本一次仅支持处理一个音频文件，请分次调用。")

    env_path = Path(args.env_file).expanduser().resolve()
    env_data = load_env_file(env_path)

    upgrade_env = env_bool(env_data, "UPGRADE_TORCH")
    upgrade_torch = upgrade_env if upgrade_env is not None else args.upgrade_torch

    torch_version = env_data.get("TORCH_VERSION") or args.torch_version
    torchvision_version = env_data.get("TORCHVISION_VERSION") or args.torchvision_version
    torchaudio_version = env_data.get("TORCHAUDIO_VERSION") or args.torchaudio_version

    torch_index_url = env_data.get("TORCH_INDEX_URL") or args.torch_index_url
    if torch_index_url is None:
        torch_index_url = "https://download.pytorch.org/whl/cu128"

    cudnn_root = env_data.get("CUDNN_ROOT") or args.cudnn_root

    if args.ssh:
        ssh_spec = args.ssh
        login = parse_ssh_command(ssh_spec)
    else:
        while True:
            ssh_spec = input(
                "请输入 SSH 连接命令 (例如 ssh -p 25810 root@connect.westc.gpuhub.com): "
            ).strip()
            if not ssh_spec:
                print("SSH 命令不能为空，请重新输入。")
                continue
            try:
                login = parse_ssh_command(ssh_spec)
                break
            except ValueError as exc:
                print(f"解析失败: {exc}")

    password = args.password
    if password is None:
        password = getpass.getpass(f"输入 {login['user']}@{login['host']} 的密码 (留空使用密钥登录): ") or None

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=login["host"],
        port=int(login["port"]),
        username=login["user"],
        password=password,
        allow_agent=True,
        look_for_keys=True,
    )
    sftp: Optional[SFTPClient] = None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stem = local_audio_paths[0].stem if local_audio_paths else "audio"
    run_name = f"whisperx-{timestamp}-{stem}"
    remote_root = (env_data.get("WHISPERX_REMOTE_DIR") or args.remote_dir).rstrip("/")
    remote_run_dir = f"{remote_root}/{run_name}"
    remote_audio_paths: List[str] = []
    upload_mapping: Dict[str, Path] = {}

    try:
        sftp = ssh.open_sftp()

        if upgrade_torch:
            upgrade_command = build_torch_upgrade_command(
                remote_project=env_data.get("WHISPERX_PROJECT_DIR") or args.remote_project,
                torch_version=torch_version,
                torchvision_version=torchvision_version,
                torchaudio_version=torchaudio_version,
                index_url=torch_index_url,
            )
            if upgrade_command:
                print(f"执行 PyTorch 升级: {upgrade_command}")
                execute_command(ssh, upgrade_command)

        ensure_remote_dirs(ssh, [remote_run_dir])

        for audio_path in local_audio_paths:
            remote_audio = f"{remote_run_dir}/{audio_path.name}"
            remote_audio_paths.append(remote_audio)
            upload_mapping[remote_audio] = audio_path

        upload_files(sftp, upload_mapping)

        diarize_flag = env_bool(env_data, "WHISPERX_DIARIZE")
        diarize = diarize_flag if diarize_flag is not None else (not args.no_diarize)

        def env_int(key: str, fallback: Optional[int]) -> Optional[int]:
            raw_val = env_data.get(key)
            if raw_val is None or raw_val.strip() == "":
                return fallback
            return int(raw_val)

        hf_endpoint = (env_data.get("HF_ENDPOINT") or "https://huggingface.co").strip() or None

        # Collect optional cache/home env vars to export on the remote before running.
        export_env: Dict[str, str] = {}
        for key in [
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
            # Download backends tuning
            "HF_HUB_DISABLE_XET",
            "HF_HUB_ENABLE_HF_TRANSFER",
        ]:
            val = env_data.get(key)
            if val is not None and val.strip() != "":
                export_env[key] = val.strip()

        # Proactively create cache directories on the remote (some plugins don't auto-create).
        cache_dirs: List[str] = []
        for key in [
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
        ]:
            v = export_env.get(key)
            if v:
                cache_dirs.append(v)
        if cache_dirs:
            ensure_remote_dirs(ssh, cache_dirs)

        remote_command = build_whisperx_command(
            remote_project=env_data.get("WHISPERX_PROJECT_DIR") or args.remote_project,
            remote_audio_paths=remote_audio_paths,
            model=env_data.get("WHISPERX_MODEL") or args.model,
            batch_size=env_int("WHISPERX_BATCH_SIZE", args.batch_size) or args.batch_size,
            compute_type=env_data.get("WHISPERX_COMPUTE_TYPE") or args.compute_type,
            output_dir=remote_run_dir,
            device=env_data.get("WHISPERX_DEVICE") or args.device,
            hf_token=env_data.get("HUGGINGFACE_API_KEY") or args.hf_token,
            language=env_data.get("WHISPERX_LANGUAGE") or args.language,
            min_speakers=env_int("WHISPERX_MIN_SPEAKERS", args.min_speakers),
            max_speakers=env_int("WHISPERX_MAX_SPEAKERS", args.max_speakers),
            diarize_model=env_data.get("WHISPERX_DIARIZE_MODEL") or args.diarize_model,
            diarize=diarize,
            hf_endpoint=hf_endpoint,
            cudnn_root=cudnn_root,
            env_vars=export_env,
        )

        print(f"执行远程命令: {remote_command}")
        log_lines = execute_command(ssh, remote_command, timeout=args.timeout or None)

        if args.local_output != DEFAULT_LOCAL_OUTPUT:
            local_output_root = Path(args.local_output).expanduser()
        elif env_data.get("WHISPERX_LOCAL_OUTPUT"):
            local_output_root = Path(env_data["WHISPERX_LOCAL_OUTPUT"]).expanduser()
        else:
            local_output_root = local_audio_paths[0].parent
        local_run_dir = local_output_root / run_name
        local_run_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_outputs(
            sftp,
            remote_run_dir,
            local_run_dir,
            skip=[path.name for path in local_audio_paths],
        )

        log_path = local_run_dir / f"{run_name}.log"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"日志已写入 {log_path}")

        # Produce a small handoff index to standardize downstream usage
        best_json: Optional[Path] = None
        best_txt: Optional[Path] = None
        # Prefer largest .txt as transcript
        txt_candidates = [p for p in downloaded if p.suffix.lower() == ".txt"]
        if txt_candidates:
            best_txt = max(txt_candidates, key=lambda p: p.stat().st_size)
        # Detect diarization JSON
        json_candidates = [p for p in downloaded if p.suffix.lower() == ".json"]
        for p in json_candidates:
            if _is_diarization_json(p):
                best_json = p
                break
        speakers_map: Dict[str, str] = {}
        if best_json is not None:
            speakers_map = _collect_speakers_from_json(best_json)
        handoff = {
            "run_name": run_name,
            "remote_run_dir": remote_run_dir,
            "local_run_dir": str(local_run_dir),
            "audio": str(local_audio_paths[0]) if local_audio_paths else None,
            "files": [
                {"name": p.name, "path": str(p), "size": p.stat().st_size}
                for p in downloaded
            ],
            "best": {
                "diarization_json": str(best_json) if best_json else None,
                "transcript_txt": str(best_txt) if best_txt else None,
            },
            "speakers": {
                "raw_to_tag": speakers_map,
                "tags": sorted(set(speakers_map.values())) if speakers_map else [],
            },
        }
        (local_run_dir / "handoff.json").write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"交接索引已写入 {(local_run_dir / 'handoff.json')}")

        if not args.keep_remote:
            cleanup_cmd = f"rm -rf {shlex.quote(remote_run_dir)}"
            ssh.exec_command(cleanup_cmd)

        print("下载的文件：")
        for file_path in downloaded:
            print(f" - {file_path}")
        print("完成。")
    finally:
        if sftp is not None:
            sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
