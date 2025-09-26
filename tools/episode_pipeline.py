"""交互式播客本地化流水线。

该脚本把 WhisperX 识别、说话人样本提取、翻译与 TTS 合成串成一个
可断点续跑、强交互的流程。所有产物统一写入项目目录，远程操作需
一次性输入 SSH 凭据，生命周期内自动复用。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import json
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import paramiko
import stat


PROJECT_BASE = Path("/Users/peiwenyang/Development/podcasts-test")
DEFAULT_REMOTE_REPO = "/root/index-tts"
# 统一的远端工作根目录，集中放置输出
DEFAULT_REMOTE_WORKDIR = "/root/autodl-fs/outputs"
DEFAULT_WHISPERX_PROJECT = "/root/autodl-fs/whisperX"
STEP_ORDER = ["transcribe", "samples", "translate", "tts_prep", "synthesize"]


class PipelineError(RuntimeError):
    pass


class StepRetry(PipelineError):
    pass


class StepAbort(PipelineError):
    pass


def ensure_project_base() -> None:
    PROJECT_BASE.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}\n")


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
        print("请输入有效内容。")


def yes_no(question: str, default: bool = True) -> bool:
    mapping = {"y": True, "n": False}
    while True:
        choice = "Y" if default else "y"
        alt = "n" if default else "N"
        raw = input(f"{question} ({choice}/{alt}): ").strip().lower()
        if not raw:
            return default
        if raw in mapping:
            return mapping[raw]
        print("请输入 y 或 n。")


def choose(prompt_text: str, options: List[str], default_index: int = 0) -> int:
    if not options:
        raise ValueError("必须提供选项。")
    for idx, item in enumerate(options, 1):
        mark = "*" if idx - 1 == default_index else " "
        print(f"  {idx}. {item}{' (默认)' if mark == '*' else ''}")
    while True:
        raw = input(f"{prompt_text}: ").strip()
        if not raw:
            return default_index
        if raw.isdigit():
            value = int(raw) - 1
            if 0 <= value < len(options):
                return value
        print("请输入有效序号。")


@dataclass
class StepInfo:
    status: str = "pending"
    updated_at: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)

    def mark(self, status: str, **meta: str) -> None:
        self.status = status
        self.updated_at = now_iso()
        if status in {"running", "pending"}:
            self.meta.clear()
        if meta:
            self.meta.update(meta)


@dataclass
class EpisodeState:
    episode_id: str
    root: Path
    audio_file: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    remote: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Dict[str, str]] = field(default_factory=dict)
    steps: Dict[str, StepInfo] = field(default_factory=lambda: {step: StepInfo() for step in STEP_ORDER})

    @property
    def state_path(self) -> Path:
        return self.root / "episode.json"

    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "created_at": self.created_at,
            "audio_file": self.audio_file,
            "remote": self.remote,
            "params": self.params,
            "steps": {
                name: {
                    "status": info.status,
                    "updated_at": info.updated_at,
                    "meta": info.meta,
                }
                for name, info in self.steps.items()
            },
        }

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, episode_id: str) -> "EpisodeState":
        root = PROJECT_BASE / episode_id
        if not root.is_dir():
            raise FileNotFoundError(f"项目目录不存在: {root}")
        path = root / "episode.json"
        if not path.is_file():
            state = cls(episode_id=episode_id, root=root)
            state.ensure_structure()
            state.save()
            return state
        data = json.loads(path.read_text(encoding="utf-8"))
        steps_data = data.get("steps", {})
        steps: Dict[str, StepInfo] = {}
        for name in STEP_ORDER:
            raw = steps_data.get(name) or {}
            steps[name] = StepInfo(
                status=raw.get("status", "pending"),
                updated_at=raw.get("updated_at"),
                meta=dict(raw.get("meta", {})),
            )
        state = cls(
            episode_id=episode_id,
            root=root,
            audio_file=data.get("audio_file"),
            created_at=data.get("created_at", now_iso()),
            remote=dict(data.get("remote", {})),
            params=dict(data.get("params", {})),
            steps=steps,
        )
        state.ensure_structure()
        return state

    def ensure_structure(self) -> None:
        for sub in [
            "raw",
            "whisperx",
            "samples",
            "translate",
            "run",
            "synth",
            "logs",
        ]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def step(self, name: str) -> StepInfo:
        return self.steps[name]

    def reset_after(self, name: str) -> None:
        if name not in STEP_ORDER:
            return
        index = STEP_ORDER.index(name)
        for step in STEP_ORDER[index:]:
            info = self.step(step)
            info.status = "pending"
            info.meta.clear()
            info.updated_at = None


@dataclass
class RemoteConfig:
    host: str
    port: int
    user: str
    password: Optional[str]
    remote_repo: str
    remote_workdir_base: str
    whisperx_project: str

    def ssh_command(self) -> str:
        if self.port != 22:
            return f"ssh -p {self.port} {self.user}@{self.host}"
        return f"{self.user}@{self.host}"


class RemoteSession:
    def __init__(self, cfg: RemoteConfig) -> None:
        self.cfg = cfg
        self._client: Optional[paramiko.SSHClient] = None
        self._lock = threading.Lock()

    def _ensure(self) -> paramiko.SSHClient:
        with self._lock:
            if self._client is not None:
                return self._client
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=self.cfg.host,
                port=self.cfg.port,
                username=self.cfg.user,
                password=self.cfg.password,
                look_for_keys=not bool(self.cfg.password),
            )
            self._client = client
            return client

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def run(self, command: str, timeout: Optional[int] = None) -> Tuple[int, List[str]]:
        client = self._ensure()
        stdin, stdout, stderr = client.exec_command(command, get_pty=True, timeout=timeout)
        output: List[str] = []
        for line in stdout:
            text = line.rstrip("\n")
            output.append(text)
            print(text)
        err_text = stderr.read().decode("utf-8", errors="ignore")
        if err_text:
            for line in err_text.rstrip("\n").splitlines():
                output.append(line)
                print(line)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, output

    def ensure_dir(self, path: str) -> None:
        self.run(f"mkdir -p {shlex.quote(path)}")

    def download_dir(self, remote_dir: str, local_dir: Path) -> None:
        client = self._ensure()
        sftp = client.open_sftp()
        try:
            self._recursive_download(sftp, remote_dir, local_dir)
        finally:
            sftp.close()

    def _recursive_download(self, sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{entry.filename}"
            local_path = local_dir / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                self._recursive_download(sftp, remote_path, local_path)
            else:
                sftp.get(remote_path, str(local_path))


class EpisodePipeline:
    def __init__(self, state: EpisodeState, remote_cfg: RemoteConfig, *, dry_run: bool = False) -> None:
        self.state = state
        self.remote_cfg = remote_cfg
        self.remote = RemoteSession(remote_cfg)
        self.dry_run = dry_run

    def close(self) -> None:
        self.remote.close()

    # ---------- 项目层操作 ----------

    def ensure_audio(self) -> Path:
        raw_dir = self.state.root / "raw"
        candidates = sorted(raw_dir.glob("*"))
        if candidates:
            if self.dry_run:
                audio = candidates[0]
            else:
                choices = [path.name for path in candidates]
                idx = choose("选择要处理的音频", choices, 0)
                audio = candidates[idx]
        else:
            while True:
                src = Path(prompt("请输入音频文件路径"))
                if src.is_file():
                    dest = raw_dir / src.name
                    shutil.copy2(src, dest)
                    audio = dest
                    break
                print("文件不存在，请重试。")
        self.state.audio_file = audio.name
        self.state.save()
        return audio

    def run(self, from_step: Optional[str] = None) -> None:
        audio_path = self.ensure_audio()
        # 起始步骤选择：支持命令行指定或交互选择
        step_names_cn = [
            "语音识别 (transcribe)",
            "样本提取 (samples)",
            "翻译优化 (translate)",
            "TTS 准备 (tts_prep)",
            "语音合成 (synthesize)",
        ]
        if from_step and from_step in STEP_ORDER:
            start_index = STEP_ORDER.index(from_step)
        else:
            # 默认从第一个非 done 的步骤开始
            start_index = 0
            for i, name in enumerate(STEP_ORDER):
                if self.state.step(name).status != "done":
                    start_index = i
                    break
            # 允许用户覆盖默认选择（dry-run 下跳过交互，直接使用默认）
            if not self.dry_run:
                print_header("选择起始步骤")
                start_index = choose("从哪一步开始重新执行?", step_names_cn, default_index=start_index)

        # 无论当前状态如何，按选择的起点将该步及其后的状态重置为 pending，确保可重新执行
        chosen_step = STEP_ORDER[start_index]
        self.state.reset_after(chosen_step)
        self.state.save()

        for step in STEP_ORDER[start_index:]:
            while True:
                info = self.state.step(step)
                if info.status == "done":
                    break
                if info.status == "failed":
                    if not self.dry_run:
                        if not yes_no(f"步骤 {step} 上次失败，是否重试?", True):
                            break
                    info.status = "pending"
                    info.meta.clear()
                    info.updated_at = None
                    self.state.save()
                elif info.status == "pending":
                    if not self.dry_run:
                        if not yes_no(f"是否执行步骤 {step}?", True):
                            print(f"跳过步骤 {step}，保持待执行状态。")
                            break

                try:
                    if step == "transcribe":
                        self.step_transcribe(audio_path)
                    elif step == "samples":
                        self.step_samples()
                    elif step == "translate":
                        self.step_translate()
                    elif step == "tts_prep":
                        self.step_prepare_tts()
                    elif step == "synthesize":
                        self.step_synthesize()
                except StepRetry as exc:
                    print(f"步骤 {step} 需要重新执行: {exc}")
                    self.state.reset_after(step)
                    self.state.save()
                    continue
                except StepAbort as exc:
                    print(f"步骤 {step} 被用户中断: {exc}")
                    info.mark("failed", error=str(exc))
                    self.state.save()
                    raise
                except PipelineError as exc:
                    print(f"步骤 {step} 失败: {exc}")
                    info.mark("failed", error=str(exc))
                    self.state.save()
                    raise
                except Exception as exc:  # noqa: BLE001
                    print(f"步骤 {step} 异常: {exc}")
                    info.mark("failed", error=str(exc))
                    self.state.save()
                    raise

                self.state.save()
                if info.status == "done":
                    break

    # ---------- 各步骤 ----------

    def step_transcribe(self, audio_path: Path) -> None:
        info = self.state.step("transcribe")
        info.mark("running")
        self.state.save()

        print_header("运行 WhisperX 远程转录")
        output_dir = self.state.root / "whisperx"
        ssh_cmd = self.remote_cfg.ssh_command()
        # Dry-run: 生成最小 handoff.json 以便后续步骤联动
        if self.dry_run:
            remote_episode_base = f"{self.remote_cfg.remote_workdir_base}/{self.state.episode_id}"
            run_name = f"whisperx-DRYRUN-{audio_path.stem}"
            remote_run_dir = f"{remote_episode_base}/whisperx/{run_name}"
            local_run_dir = output_dir / run_name
            local_run_dir.mkdir(parents=True, exist_ok=True)
            handoff = {
                "run_name": run_name,
                "remote_run_dir": remote_run_dir,
                "local_run_dir": str(local_run_dir),
                "audio": str(audio_path),
                "files": [],
                "best": {
                    "diarization_json": f"{remote_run_dir}/{audio_path.stem}.json",
                    "transcript_txt": f"{remote_run_dir}/{audio_path.stem}.txt",
                },
                "speakers": {
                    "raw_to_tag": {"SPEAKER_00": "speaker1", "SPEAKER_01": "speaker2"},
                    "tags": ["speaker1", "speaker2"],
                },
            }
            (local_run_dir / "handoff.json").write_text(
                json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            info.mark(
                "done",
                run_name=run_name,
                local_run=str(local_run_dir),
                remote_run=remote_run_dir,
                diar_json=handoff["best"]["diarization_json"],
                transcript=handoff["best"]["transcript_txt"],
            )
            self.state.remote["whisperx_run_dir"] = remote_run_dir
            self.state.remote["audio_remote"] = self._guess_remote_audio_path(handoff)
            self.state.save()
            print("[dry-run] WhisperX 完成 (模拟)。")
            return
        # 按项目名聚合远程目录: /root/autodl-fs/outputs/<episode>/whisperx
        remote_episode_base = f"{self.remote_cfg.remote_workdir_base}/{self.state.episode_id}"
        cmd = [
            sys.executable,
            "tools/whisperx_runner.py",
            str(audio_path),
            "--env-file",
            str(Path("tools/whisperx.env")),
            "--ssh",
            ssh_cmd,
            "--keep-remote",
            "--local-output",
            str(output_dir),
            "--remote-dir",
            f"{remote_episode_base}/whisperx",
            "--remote-project",
            self.remote_cfg.whisperx_project or DEFAULT_WHISPERX_PROJECT,
        ]
        if self.remote_cfg.password:
            cmd.extend(["--password", self.remote_cfg.password])
        print("执行命令:")
        print(" ".join(shlex.quote(part) for part in cmd))
        result = subprocess.run(cmd, cwd=str(Path.cwd()), text=True)
        if result.returncode != 0:
            raise PipelineError("WhisperX 运行失败")

        handoff = self._load_latest_handoff()
        info.mark(
            "done",
            run_name=handoff.get("run_name", ""),
            local_run=handoff.get("local_run_dir", ""),
            remote_run=handoff.get("remote_run_dir", ""),
            diar_json=handoff.get("best", {}).get("diarization_json", ""),
            transcript=handoff.get("best", {}).get("transcript_txt", ""),
        )
        self.state.remote["whisperx_run_dir"] = handoff.get("remote_run_dir", "")
        self.state.remote["audio_remote"] = self._guess_remote_audio_path(handoff)
        self.state.save()
        print("WhisperX 完成。")

    def _load_latest_handoff(self) -> Dict:
        whisperx_dir = self.state.root / "whisperx"
        handoffs = list(whisperx_dir.rglob("handoff.json"))
        if not handoffs:
            raise PipelineError("未找到 handoff.json")
        handoffs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        data = json.loads(handoffs[0].read_text(encoding="utf-8"))
        return data

    def _guess_remote_audio_path(self, handoff: Dict) -> str:
        remote_run = handoff.get("remote_run_dir") or ""
        audio_path = handoff.get("audio")
        if remote_run and audio_path:
            return f"{remote_run}/{Path(audio_path).name}"
        return ""

    def step_samples(self) -> None:
        info = self.state.step("samples")
        info.mark("running")
        self.state.save()

        local_handoff = self._load_latest_handoff()
        diar_json_local = Path(local_handoff.get("best", {}).get("diarization_json") or "")
        audio_local = Path(local_handoff.get("audio") or "")
        if not diar_json_local.is_file() or not audio_local.is_file():
            raise PipelineError("缺少本地 WhisperX 产物 (音频/JSON)")

        local_samples = self.state.root / "samples"
        if local_samples.exists():
            for entry in local_samples.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

        preset = self.state.params.get("samples", {}).get("preset", "podcast")
        per_speaker = int(self.state.params.get("samples", {}).get("per_speaker", "1"))
        if not self.dry_run and yes_no(f"样本提取将使用 preset={preset}, 每人 {per_speaker} 段，是否调整?", False):
            preset = prompt("请输入 preset (podcast/strict/relaxed/fast)", preset)
            per_speaker = int(prompt("请输入每位说话人的样本数量", str(per_speaker)))
        self.state.params.setdefault("samples", {})["preset"] = preset
        self.state.params["samples"]["per_speaker"] = str(per_speaker)
        self.state.save()

        # Dry-run: 生成本地样本与 manifest
        if self.dry_run:
            from shutil import which
            local_samples.mkdir(parents=True, exist_ok=True)
            spk1 = local_samples / "speaker1.mp3"
            spk2 = local_samples / "speaker2.mp3"
            if which("ffmpeg"):
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", "1", "-c:a", "libmp3lame", "-q:a", "7", str(spk1)
                ], check=True)
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", "1", "-c:a", "libmp3lame", "-q:a", "7", str(spk2)
                ], check=True)
            else:
                spk1.write_bytes(b"")
                spk2.write_bytes(b"")
            manifest = {
                "audio": self.state.remote.get("audio_remote", ""),
                "metadata": remote_json,
                "output_dir": str(local_samples),
                "clips": [],
                "summary": {
                    "selected": {"speaker1": [str(spk1)], "speaker2": [str(spk2)]},
                    "alternatives": {},
                    "speakers": ["speaker1", "speaker2"],
                },
            }
            (local_samples / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.state.remote["samples_dir"] = remote_samples_dir
            info.mark("done", manifest=str(local_samples / "manifest.json"))
            self.state.save()
            print("[dry-run] 样本生成完成 (模拟)。")
            return

        # 启动并发：本地翻译（不依赖已选样本，仅按说话人映射改写标签）
        # 读取说话人映射以便翻译脚本中统一使用 speakerX 标签
        spk_map = {}
        try:
            spk_map = dict(local_handoff.get("speakers", {}).get("raw_to_tag", {}))
        except Exception:
            spk_map = {}

        translate_thread: Optional[threading.Thread] = None
        translate_ok = {"ok": False, "script": None, "analysis": None}

        def _run_local_translate_background() -> None:
            # 目标目录: <episode>/translate
            local_translate = self.state.root / "translate"
            local_translate.mkdir(parents=True, exist_ok=True)
            for entry in list(local_translate.iterdir()):
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            transcript_local = local_handoff.get("best", {}).get("transcript_txt")
            if not transcript_local or not Path(transcript_local).is_file():
                # 极端情况：按远端命名惯例在本地查找
                transcript_local = str(self.state.root / "whisperx" / Path(diar_json_local).with_suffix(".txt").name)
            cmd_local: List[str] = [
                sys.executable,
                "tools/translate_openai_cli.py",
                "--env",
                "tools/openai_translator.env",
                "--input",
                str(transcript_local),
                "--output",
                str(local_translate / "translation.json"),
                "--emit-json",
                "--json-output",
                str(local_translate / "translation.json"),
                "--script-out",
                str(local_translate / "script.txt"),
                "--analysis-out",
                str(local_translate / "analysis.md"),
            ]
            # 翻译参数（与 step_translate 保持一致）
            params = self.state.params.setdefault("translate", {})
            temperature = params.get("temperature", "0.2")
            model = params.get("model", "")
            cmd_local.extend(["--temperature", str(temperature)])
            if model:
                cmd_local.extend(["--model", str(model)])
            # 统一标签
            for old, new in spk_map.items():
                cmd_local.extend(["--rewrite-speaker", f"{old}={new}"])
            print("并发执行本地翻译:")
            print(" ".join(shlex.quote(p) for p in cmd_local))
            res = subprocess.run(cmd_local, cwd=str(Path.cwd()), text=True)
            if res.returncode == 0:
                translate_ok["ok"] = True
                translate_ok["script"] = str(local_translate / "script.txt")
                translate_ok["analysis"] = str(local_translate / "analysis.md")

        # 仅在非 dry-run 下并发
        if not self.dry_run:
            translate_thread = threading.Thread(target=_run_local_translate_background, daemon=True)
            translate_thread.start()

        # 本地执行样本提取（首次尝试）
        local_samples.mkdir(parents=True, exist_ok=True)
        manifest_path = local_samples / "manifest.json"
        cmd_extract = [
            sys.executable,
            "tools/extract_speaker_samples.py",
            str(audio_local),
            str(diar_json_local),
            "--output-dir",
            str(local_samples),
            "--manifest",
            str(manifest_path),
            "--preset",
            str(preset),
            "--per-speaker",
            str(per_speaker),
            "--workers",
            "4",
        ]
        print("本地执行样本提取:")
        print(" ".join(shlex.quote(p) for p in cmd_extract))
        res1 = subprocess.run(cmd_extract, cwd=str(Path.cwd()), text=True)
        if res1.returncode != 0 or not manifest_path.is_file():
            raise PipelineError("本地样本提取失败")

        summary = self._load_samples_summary(manifest_path)
        if not (summary.get("speakers") or []):
            print("[warn] 首次样本提取未找到可用片段，将调用 Codex 提供调参建议后重试…")
            # 使用 Codex 辅助调参并重试
            cmd_extract_codex = cmd_extract + [
                "--assistant-engine",
                "codex",
                "--assistant-codex-flags",
                "--full-auto",
                "--assistant-timeout",
                "300",
                "--assistant-log",
                str(local_samples / "assistant_log.json"),
            ]
            print("Codex 辅助样本提取:")
            print(" ".join(shlex.quote(p) for p in cmd_extract_codex))
            res2 = subprocess.run(cmd_extract_codex, cwd=str(Path.cwd()), text=True)
            if res2.returncode != 0 or not manifest_path.is_file():
                print("[warn] Codex 重试失败，尝试使用宽松参数再试一次…")
                cmd_relaxed = cmd_extract + [
                    "--min-start",
                    "0",
                    "--skip-head",
                    "0",
                    "--hard-skip-tail",
                    "0",
                    "--min-speech-ratio",
                    "0.6",
                    "--bgm-threshold",
                    "0.7",
                    "--preset",
                    "relaxed",
                ]
                res3 = subprocess.run(cmd_relaxed, cwd=str(Path.cwd()), text=True)
                if res3.returncode != 0 or not manifest_path.is_file():
                    raise PipelineError("样本提取在多次尝试后仍失败")
            # 重新读取 summary
            summary = self._load_samples_summary(manifest_path)

        # 等待并发翻译完成（若仍在进行）
        if translate_thread is not None:
            translate_thread.join(timeout=5 * 60)

        # 标记样本步骤完成并确认
        self.state.remote["samples_dir"] = str(local_samples)
        info.mark("done", manifest=str(manifest_path))
        self.state.save()
        self._confirm_samples(summary, local_samples)
        # 翻译由下一步进行确认；若并发已生成译稿，step_translate 将自动复用

    def _load_samples_summary(self, manifest_path: Path) -> Dict:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = data.get("summary") or {}
        return summary

    def _confirm_samples(self, summary: Dict, local_samples: Path) -> None:
        speakers = summary.get("speakers") or []
        selected = summary.get("selected") or {}
        if not speakers:
            print("未在 summary 中找到说话人列表，默认接受。")
            return

        print_header("样本确认")
        for spk in speakers:
            files = selected.get(spk) or []
            rel_files = ", ".join(Path(f).name for f in files)
            print(f"- {spk}: {rel_files}")
        if yes_no("是否接受当前样本?", True):
            print("样本已确认。")
            return

        if yes_no("是否重新生成样本?", True):
            raise StepRetry("用户要求重新提取样本")
        print("保留现有样本。")

    def step_translate(self) -> None:
        info = self.state.step("translate")
        info.mark("running")
        self.state.save()

        # 如果并发阶段已在本地生成译稿，直接进入确认环节
        pre_local = self.state.root / "translate"
        pre_script = pre_local / "script.txt"
        pre_analysis = pre_local / "analysis.md"
        if pre_script.is_file():
            info.mark("done", script=str(pre_script))
            self.state.save()
            self._confirm_translation(pre_script, pre_analysis)
            return

        handoff = self._load_latest_handoff()
        transcript = handoff.get("best", {}).get("transcript_txt")
        remote_run = self.state.remote.get("whisperx_run_dir")
        remote_samples_dir = self.state.remote.get("samples_dir")
        if not transcript or not remote_run or not remote_samples_dir:
            raise PipelineError("缺少翻译所需的远程文件信息")

        remote_input = f"{remote_run}/{Path(transcript).name}"
        # 统一到项目根目录下: /root/autodl-fs/outputs/<episode>/translate
        remote_episode_base = f"{self.remote_cfg.remote_workdir_base}/{self.state.episode_id}"
        remote_out_dir = f"{remote_episode_base}/translate"
        code = 0
        if not self.dry_run:
            code, _ = self.remote.run(
                f"rm -rf {shlex.quote(remote_out_dir)} && mkdir -p {shlex.quote(remote_out_dir)}"
            )
            if code != 0:
                print("[warn] 远程翻译目录准备失败，将尝试本地翻译。")

        params = self.state.params.setdefault("translate", {})
        temperature = params.get("temperature", "0.2")
        model = params.get("model", "")
        if (not self.dry_run) and yes_no(f"翻译将使用 temperature={temperature}{', model='+model if model else ''}，是否调整?", False):
            temperature = prompt("请输入 temperature", temperature)
            model = prompt("请输入模型名称 (留空使用默认)", model, allow_empty=True)
        params["temperature"] = temperature
        params["model"] = model
        self.state.save()

        # Dry-run: 生成本地脚本与分析文件
        if self.dry_run:
            local_translate = self.state.root / "translate"
            local_translate.mkdir(parents=True, exist_ok=True)
            (local_translate / "script.txt").write_text(
                "[speaker1] 这是测试脚本。\n[speaker2] 你好，世界。\n",
                encoding="utf-8",
            )
            (local_translate / "analysis.md").write_text(
                "# 测试分析\n\n此为 dry-run 生成的占位分析。\n",
                encoding="utf-8",
            )
            info.mark("done", script=str(local_translate / "script.txt"))
            self.state.save()
            print("[dry-run] 翻译完成 (模拟)。")
            return

        prefix = (
            f"cd {shlex.quote(self.remote_cfg.remote_repo)} && "
            # 确保 uv 在 PATH 中（覆盖常见安装路径）
            "export PATH=\"$HOME/.local/bin:/usr/local/bin:/root/miniconda3/bin:/opt/conda/bin:$PATH\" && "
            "(source /etc/network_turbo >/dev/null 2>&1 || true)"
        )
        command = (
            f"{prefix} && uv run python tools/translate_openai_cli.py "
            "--env tools/openai_translator.env "
            f"--input {shlex.quote(remote_input)} "
            f"--output {shlex.quote(remote_out_dir + '/translation.json')} "
            "--emit-json "
            f"--json-output {shlex.quote(remote_out_dir + '/translation.json')} "
            f"--script-out {shlex.quote(remote_out_dir + '/script.txt')} "
            f"--analysis-out {shlex.quote(remote_out_dir + '/analysis.md')} "
            f"--ensure-speakers-from {shlex.quote(remote_samples_dir)} "
            f"--temperature {shlex.quote(temperature)}"
        )
        if model:
            command += f" --model {shlex.quote(model)}"
        def do_local_translate() -> Tuple[Path, Path]:
            local_translate = self.state.root / "translate"
            local_translate.mkdir(parents=True, exist_ok=True)
            # 清空旧产物
            for entry in list(local_translate.iterdir()):
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            # 优先使用本地 handoff 的 transcript
            local_transcript = handoff.get("best", {}).get("transcript_txt")
            if not local_transcript or not Path(local_transcript).is_file():
                # 回退为从远端下载目录构造的路径名（极端情况下）
                local_transcript = str(self.state.root / "whisperx" / Path(remote_input).name)
            cmd_local = [
                sys.executable,
                "tools/translate_openai_cli.py",
                "--env",
                "tools/openai_translator.env",
                "--input",
                local_transcript,
                "--output",
                str(local_translate / "translation.json"),
                "--emit-json",
                "--json-output",
                str(local_translate / "translation.json"),
                "--script-out",
                str(local_translate / "script.txt"),
                "--analysis-out",
                str(local_translate / "analysis.md"),
                "--ensure-speakers-from",
                str(self.state.root / "samples"),
            ]
            print("改为本地执行翻译:")
            print(" ".join(shlex.quote(p) for p in cmd_local))
            res = subprocess.run(cmd_local, cwd=str(Path.cwd()), text=True)
            if res.returncode != 0:
                raise PipelineError("本地翻译失败")
            return local_translate / "script.txt", local_translate / "analysis.md"

        try:
            print("远程执行翻译:")
            print(command)
            exit_code, _ = self.remote.run(command, timeout=1800)
            if exit_code != 0:
                raise PipelineError("远程翻译失败")
            local_translate = self.state.root / "translate"
            local_translate.mkdir(parents=True, exist_ok=True)
            for entry in list(local_translate.iterdir()):
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            self.remote.download_dir(remote_out_dir, local_translate)
            script_path = local_translate / "script.txt"
            analysis_path = local_translate / "analysis.md"
            if not script_path.is_file():
                raise PipelineError("翻译脚本缺失")
        except Exception as exc:
            print(f"[warn] 远程翻译失败 ({exc})，将尝试在本地执行。")
            script_path, analysis_path = do_local_translate()

        info.mark("done", script=str(script_path))
        self.state.save()
        self._confirm_translation(script_path, analysis_path)

    def _confirm_translation(self, script_path: Path, analysis_path: Path) -> None:
        print_header("翻译确认")
        if analysis_path.is_file():
            content = analysis_path.read_text(encoding="utf-8")
            snippet = "\n".join(content.splitlines()[:20])
            print("分析摘要:")
            print(textwrap.indent(snippet, "  "))
        else:
            print("未找到分析文件。")

        while True:
            choice = choose(
                "请选择操作",
                [
                    "接受译稿",
                    "查看/修改 script.txt 后继续",
                    "重新翻译",
                    "退出流程",
                ],
            )
            if choice == 0:
                print("译稿已确认。")
                return
            if choice == 1:
                print(f"请在另一个窗口中编辑: {script_path}")
                input("修改完成后按回车继续...")
                continue
            if choice == 2:
                raise StepRetry("用户要求重新翻译")
            raise StepAbort("用户中断流程")

    def step_prepare_tts(self) -> None:
        info = self.state.step("tts_prep")
        info.mark("running")
        self.state.save()

        manifest_path = self.state.root / "samples" / "manifest.json"
        script_path = self.state.root / "translate" / "script.txt"
        if not manifest_path.is_file() or not script_path.is_file():
            raise PipelineError("缺少样本或译稿")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = manifest.get("summary") or {}
        selected = summary.get("selected") or {}
        run_dir = self.state.root / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        # 清理旧文件
        for f in run_dir.glob("*"):
            if f.is_file():
                f.unlink()

        missing = []
        for speaker, files in selected.items():
            if not files:
                missing.append(speaker)
                continue
            src = Path(files[0])
            if not src.is_absolute():
                src = (self.state.root / "samples" / Path(files[0]).name)
            dest = run_dir / f"{speaker}.mp3"
            if not src.is_file():
                missing.append(speaker)
                continue
            shutil.copy2(src, dest)

        script_dest = run_dir / f"{self.state.episode_id}.txt"
        script_dest.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")

        if missing:
            print("以下说话人未找到样本:", ", ".join(missing))
            raise PipelineError("样本缺失，无法继续")

        info.mark("done", run_dir=str(run_dir))
        self.state.save()
        print("TTS 输入已准备。")

    def step_synthesize(self) -> None:
        info = self.state.step("synthesize")
        info.mark("running")
        self.state.save()

        run_dir = self.state.root / "run"
        if not run_dir.exists():
            raise PipelineError("缺少 run 目录")

        cmd = [
            sys.executable,
            "tools/remote_runner.py",
            "--source-dir",
            str(run_dir),
            "--story-name",
            self.state.episode_id,
            "--remote-workdir",
            f"{self.remote_cfg.remote_workdir_base}/{self.state.episode_id}",
            "--remote-repo",
            self.remote_cfg.remote_repo,
            "--local-output",
            str(self.state.root / "synth"),
            "--num-workers",
            "3",
            "--ssh",
            self.remote_cfg.ssh_command(),
        ]
        if self.dry_run:
            print("[dry-run] 远程合成命令:")
            print(" ".join(shlex.quote(part) for part in cmd))
            synth_dir = self.state.root / "synth"
            synth_dir.mkdir(parents=True, exist_ok=True)
            # 生成一段 1s 的静音占位音频
            from shutil import which
            dummy = synth_dir / f"{self.state.episode_id}.wav"
            if which("ffmpeg"):
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", "1", "-c:a", "pcm_s16le", str(dummy)
                ], check=True)
            else:
                dummy.write_bytes(b"")
            outputs = [p.name for p in synth_dir.glob("*") if p.is_file()]
            info.mark("done", outputs=", ".join(outputs))
            self.state.save()
            print("[dry-run] TTS 合成完成 (模拟)。")
            return
        if self.remote_cfg.password:
            cmd.extend(["--password", self.remote_cfg.password])
        print("执行远程合成:")
        print(" ".join(shlex.quote(part) for part in cmd))
        result = subprocess.run(cmd, cwd=str(Path.cwd()), text=True)
        if result.returncode != 0:
            raise PipelineError("TTS 合成失败")

        synth_dir = self.state.root / "synth"
        outputs = [p.name for p in synth_dir.glob("*") if p.is_file()]
        info.mark("done", outputs=", ".join(outputs))
        self.state.save()
        print("TTS 合成完成。")


def prompt_remote_config() -> RemoteConfig:
    print_header("配置远程连接")
    while True:
        # 仅需粘贴 SSH 指令（或逐项输入），其它目录固定为默认值
        ssh_raw = prompt(
            "SSH 登录指令（如: ssh -p 34672 root@example.com，可留空）",
            default="",
            allow_empty=True,
        )

        host = ""
        user = ""
        port = 22
        if ssh_raw:
            try:
                tokens = shlex.split(ssh_raw)
                if tokens and tokens[0] == "ssh":
                    tokens = tokens[1:]
                idx = 0
                while idx < len(tokens):
                    t = tokens[idx]
                    if t == "-p":
                        idx += 1
                        if idx >= len(tokens):
                            raise ValueError("-p 后缺少端口号。")
                        port = int(tokens[idx])
                    elif t.startswith("-"):
                        pass  # 其它参数忽略（容错）
                    else:
                        spec = t
                        if "@" in spec:
                            user, host = spec.split("@", 1)
                        else:
                            host = spec
                    idx += 1
                if not host:
                    raise ValueError("未解析到目标主机。")
                if not user:
                    user = getpass.getuser()
            except Exception as exc:  # noqa: BLE001
                print(f"无法解析 SSH 指令: {exc}")
                continue
        else:
            host_raw = prompt("远程主机 (可含端口，如 example.com:22774)")
            if ":" in host_raw:
                host, port_raw = host_raw.split(":", 1)
                try:
                    port = int(port_raw)
                except ValueError:
                    print("端口需为数字。")
                    continue
            else:
                host = host_raw
            user = prompt("用户名", getpass.getuser())

        password = getpass.getpass("密码 (使用密钥登录可留空): ") or None
        cfg = RemoteConfig(
            host=host,
            port=port,
            user=user,
            password=password,
            remote_repo=DEFAULT_REMOTE_REPO,
            remote_workdir_base=DEFAULT_REMOTE_WORKDIR,
            whisperx_project=DEFAULT_WHISPERX_PROJECT,
        )
        # 提示一次总览，默认确认
        print(
            f"\n远程主机: {host}:{port} (用户 {user})\n"
            f"远程仓库路径: {DEFAULT_REMOTE_REPO}\n"
            f"远程工作根目录: {DEFAULT_REMOTE_WORKDIR}\n"
            f"WhisperX 项目目录: {DEFAULT_WHISPERX_PROJECT}\n"
        )
        if yes_no("以上配置是否正确?", True):
            return cfg


def select_episode() -> EpisodeState:
    ensure_project_base()
    existing = sorted(p.name for p in PROJECT_BASE.iterdir() if p.is_dir())
    menu = ["创建新项目"] + [f"继续 {name}" for name in existing]
    idx = choose("请选择项目", menu)
    if idx == 0:
        while True:
            ep_id = prompt("请输入项目 ID (仅字母数字下划线)")
            if not ep_id:
                continue
            target = PROJECT_BASE / ep_id
            if target.exists():
                if yes_no("目录已存在，是否继续使用?", False):
                    break
                continue
            target.mkdir(parents=True, exist_ok=True)
            break
    else:
        ep_id = existing[idx - 1]
    state = EpisodeState.load(ep_id)
    return state


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="播客本地化流水线 (交互式)")
    parser.add_argument("--episode", help="直接指定项目 ID", default=None)
    parser.add_argument("--from-step", choices=STEP_ORDER, help="从指定步骤开始执行 (transcribe/samples/translate/tts_prep/synthesize)")
    parser.add_argument("--dry-run", action="store_true", help="仅本地演练，不进行远程执行，生成占位产物")
    args = parser.parse_args(argv)

    if args.episode:
        ensure_project_base()
        state = EpisodeState.load(args.episode)
    else:
        state = select_episode()

    if args.dry_run:
        remote_cfg = RemoteConfig(
            host="localhost",
            port=22,
            user=getpass.getuser(),
            password=None,
            remote_repo=DEFAULT_REMOTE_REPO,
            remote_workdir_base=DEFAULT_REMOTE_WORKDIR,
            whisperx_project=DEFAULT_WHISPERX_PROJECT,
        )
    else:
        remote_cfg = prompt_remote_config()
    pipeline = EpisodePipeline(state, remote_cfg, dry_run=bool(args.dry_run))
    try:
        pipeline.run(from_step=args.from_step)
    finally:
        pipeline.close()
    print("流程结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
