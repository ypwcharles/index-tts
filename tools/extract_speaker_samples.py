"""Utility to extract clean speaker samples from WhisperX diarization output.

This script loads the WhisperX JSON (word-level timestamps with speaker labels),
clusters consecutive words by speaker, scores each candidate segment, and exports
high-quality samples via ffmpeg. It also writes a manifest describing selected
clips and alternatives so they can be reviewed or re-generated later.
"""

from __future__ import annotations

import argparse
import sys
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
import subprocess as _subprocess
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random
import concurrent.futures as _futures
import tempfile
import shutil

import numpy as np

try:  # librosa 属于依赖列表，缺失时给出友好提示
    import librosa
except ImportError as exc:  # pragma: no cover - 运行环境缺少依赖时提示
    raise SystemExit(
        "运行该脚本需要安装 librosa，请执行 `uv sync` 或 `pip install librosa` 后重试"
    ) from exc

# 可选依赖：更稳健的 VAD
try:  # pragma: no cover - 可选依赖
    import webrtcvad  # type: ignore
except Exception:  # noqa: BLE001 - 允许任意导入失败
    webrtcvad = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speaker samples from WhisperX diarization output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="Path to the source audio file")
    parser.add_argument("metadata", type=Path, help="Path to the WhisperX JSON metadata file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store exported clips")
    parser.add_argument("--manifest", type=Path, help="Optional manifest JSON for downstream tooling")
    parser.add_argument("--min-duration", type=float, default=6.0, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=float, default=12.0, help="Maximum clip duration in seconds")
    parser.add_argument("--target-duration", type=float, default=8.0, help="Ideal clip duration used in scoring")
    parser.add_argument("--max-gap", type=float, default=0.7, help="Maximum silence gap between words when merging")
    parser.add_argument(
        "--min-start",
        type=float,
        default=300.0,
        help="强制跳过音频起始的若干秒（默认 300 秒，用于避开片头/广告；可设为 0 关闭）",
    )
    parser.add_argument(
        "--skip-head",
        type=float,
        default=300.0,
        help="De-prioritise segments that start within the first N seconds of audio",
    )
    parser.add_argument(
        "--skip-tail",
        type=float,
        default=180.0,
        help="De-prioritise segments that end within the last N seconds of audio",
    )
    parser.add_argument(
        "--hard-skip-tail",
        type=float,
        default=60.0,
        help="强制丢弃音频最后 N 秒内的片段(默认 60；设为 0 关闭)",
    )
    parser.add_argument("--per-speaker", type=int, default=1, help="Number of clips to export per speaker")
    parser.add_argument(
        "--min-speech-ratio",
        type=float,
        default=0.70,
        help="最小语音占比 (0-1)，过滤含大量音乐/背景的片段",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="WebRTC VAD aggressiveness (0-3)，越大越严格",
    )
    parser.add_argument(
        "--bgm-threshold",
        type=float,
        default=0.55,
        help="背景音乐评分阈值，超过则标记为 noisy 并降权",
    )
    parser.add_argument(
        "--extra-candidates",
        type=int,
        default=2,
        help="How many additional candidates to keep per speaker in the manifest",
    )
    parser.add_argument(
        "--tail-silence",
        type=float,
        default=1.0,
        help="Seconds of silence to append to the end of each exported clip",
    )
    parser.add_argument("--sample-rate", type=int, default=44100, help="Output sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Output channel count")
    parser.add_argument("--codec", default="libmp3lame", help="Audio codec to use for ffmpeg")
    parser.add_argument(
        "--afilter",
        default=None,
        help="自定义 ffmpeg -af 过滤器（会在末尾追加 apad）",
    )
    parser.add_argument("--bitrate", default="192k", help="Target audio bitrate")
    parser.add_argument("--dry-run", action="store_true", help="Analyse only, do not export audio")
    parser.add_argument("--interactive", action="store_true", help="Require confirmation before exporting each clip")
    parser.add_argument(
        "--allow-out-of-range",
        action="store_true",
        help="Allow exporting clips outside the duration bounds if no valid candidate exists",
    )
    parser.add_argument("--verbose", action="store_true", help="Print ffmpeg output and extra diagnostics")
    # 智能 CLI 介入：允许外部 CLI（如 Codex/Gemini）在失败时给出参数建议
    parser.add_argument(
        "--assistant-cmd",
        help="当未选出样本时调用的外部命令，接收 JSON 上下文（stdin）并输出新参数 JSON（stdout）",
    )
    parser.add_argument(
        "--assistant-engine",
        choices=["none", "external", "codex"],
        default="none",
        help="智能助手模式：none(关闭)、external(使用 --assistant-cmd)、codex(使用 Codex CLI)",
    )
    parser.add_argument(
        "--assistant-codex-flags",
        default="--full-auto",
        help="传递给 codex exec 的额外标志，例如: '--full-auto' 或 '--dangerously-bypass-approvals-and-sandbox'",
    )
    parser.add_argument(
        "--assistant-codex-model",
        default=None,
        help="可选：指定 Codex 模型名 (-m)。例如 gpt-4.1 或 gpt-5",
    )
    parser.add_argument(
        "--assistant-codex-workdir",
        type=Path,
        default=None,
        help="运行 codex 的工作目录 (-C)。默认自动检测为仓库根目录或当前工作目录",
    )
    parser.add_argument(
        "--assistant-timeout",
        type=int,
        default=180,
        help="外部助手命令超时时间（秒）",
    )
    parser.add_argument(
        "--assistant-log",
        type=Path,
        help="记录助手交互的 JSON（请求/响应）便于审计与复现",
    )
    # 性能与再生成控制
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行分析候选的工作线程数(0 为串行)",
    )
    parser.add_argument(
        "--edge-trim-max",
        type=float,
        default=0.2,
        help="导出前对首尾静音的最大内缩秒数",
    )
    parser.add_argument(
        "--avoid-manifest",
        nargs="*",
        default=None,
        help="历史清单文件(一个或多个)，避免与其中片段重叠",
    )
    parser.add_argument(
        "--avoid-margin",
        type=float,
        default=1.5,
        help="避免区间的扩展边距(秒)，扩大历史片段以避免近邻",
    )
    parser.add_argument(
        "--avoid-min-overlap",
        type=float,
        default=0.5,
        help="判定为重叠所需的最小重叠时长(秒)",
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=0,
        help="每个说话人从得分 Top-K 中随机选择(0 表示严格按分数排序)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="随机种子；与 --sample-top-k 配合用于可复现抽样",
    )
    parser.add_argument(
        "--preload-audio",
        action="store_true",
        help="预先以 16kHz 单声道载入整段音频以加速分析(占用较多内存)",
    )
    parser.add_argument(
        "--preset",
        choices=["podcast", "strict", "relaxed", "fast"],
        help="一键使用预设参数：podcast(推荐)、strict(更干净)、relaxed(更宽松)、fast(偏速度)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="可选的 JSON 配置文件，键为参数名(如 min_start)，值为期望设置；与 --preset 配合更灵活",
    )
    return parser.parse_args()


def ffprobe_duration(path: Path) -> Optional[float]:
    """Return the duration of *path* in seconds using ffprobe, if available."""

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class Candidate:
    speaker: str
    start: float
    end: float
    duration: float
    avg_conf: float
    word_count: int
    text: str
    words: List[dict]
    score_components: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"

    @property
    def score(self) -> float:
        return self.score_components.get("total", 0.0)


def load_metadata(path: Path) -> Dict[str, List[dict]]:
    """Load the WhisperX JSON and group word-level entries by speaker."""

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    speakers: Dict[str, List[dict]] = {}
    for segment in data.get("segments", []):
        for word in segment.get("words", []):
            if "speaker" not in word or word.get("end") is None:
                continue
            speakers.setdefault(word["speaker"], []).append(word)

    for items in speakers.values():
        items.sort(key=lambda w: w["start"])
    return speakers


def merge_words(words: Sequence[dict], max_gap: float) -> Iterable[List[dict]]:
    """Merge consecutive words while the silence gap is below *max_gap*."""

    if not words:
        return
    buffer = [words[0]]
    for previous, current in zip(words, words[1:]):
        if current["start"] - previous["end"] <= max_gap:
            buffer.append(current)
        else:
            yield buffer
            buffer = [current]
    if buffer:
        yield buffer


def build_candidates(
    speakers: Dict[str, List[dict]],
    *,
    max_gap: float,
    min_duration: float,
    max_duration: float,
    min_start: Optional[float],
) -> Dict[str, List[Candidate]]:
    """Create raw candidate segments per speaker."""

    candidates: Dict[str, List[Candidate]] = {}
    for speaker, words in speakers.items():
        entries: List[Candidate] = []
        for chunk in merge_words(words, max_gap):
            stack = [chunk]
            refined: List[List[dict]] = []
            while stack:
                current = stack.pop()
                cur_start = current[0]["start"]
                cur_end = current[-1]["end"]
                cur_duration = cur_end - cur_start
                if cur_duration <= max_duration * 1.2 or len(current) <= 1:
                    refined.append(current)
                    continue

                # 从中间寻找切分点（优先选择更大的词间停顿/接近目标）
                target = cur_start + min(max_duration, cur_duration / 2)
                best_idx = None
                best_score = -1.0
                for idx in range(1, len(current)):
                    prev_end = current[idx - 1]["end"]
                    cur_start_i = current[idx]["start"]
                    gap = max(0.0, cur_start_i - prev_end)
                    # 该分割点的时间
                    split_t = current[idx - 1]["end"]
                    # 距离目标越近越好
                    dist = abs(split_t - target)
                    # 以 gap 为主，鼓励接近目标
                    score = gap - 0.15 * dist
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                split_idx = best_idx or (len(current) // 2)
                left = current[:split_idx]
                right = current[split_idx:]
                if left:
                    stack.append(left)
                if right:
                    stack.append(right)

            for chunk_words in refined:
                start = chunk_words[0]["start"]
                end = chunk_words[-1]["end"]
                duration = end - start
                if duration < min_duration * 0.6:
                    continue
                avg_conf = mean(w.get("score", 1.0) for w in chunk_words)
                text = " ".join(w.get("word", "") for w in chunk_words)
                if min_start is not None and start < min_start:
                    continue
                entries.append(
                    Candidate(
                        speaker=speaker,
                        start=start,
                        end=end,
                        duration=duration,
                        avg_conf=avg_conf,
                        word_count=len(chunk_words),
                        text=text,
                        words=list(chunk_words),
                    )
                )
        candidates[speaker] = entries
    return candidates


def _load_audio_window(
    audio_path: Path,
    *,
    start: float,
    end: float,
    sr: int,
    margin: float,
) -> tuple[np.ndarray, int, float]:
    """Load a mono audio window around ``start``-``end`` with optional margins."""

    window_start = max(0.0, start - margin)
    duration = (end - start) + 2 * margin
    try:
        y, loaded_sr = librosa.load(
            audio_path,
            sr=sr,
            mono=True,
            offset=window_start,
            duration=max(duration, 0.01),
        )
    except Exception as exc:  # pragma: no cover - I/O 或解码异常
        raise RuntimeError(f"加载音频 {audio_path} 失败: {exc}") from exc

    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0)

    return y, loaded_sr, window_start


def _speech_ratio_webrtc(segment: np.ndarray, sr: int, aggressiveness: int = 2) -> float:
    """Return speech frame ratio using WebRTC VAD if available, else -1.

    WebRTC VAD requires 16-bit PCM, mono, sample rates of 8k/16k/32k/48k, and 10/20/30ms frames.
    """
    if webrtcvad is None:
        return -1.0
    if sr not in (8000, 16000, 32000, 48000):
        return -1.0

    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 20
    frame_len = int(sr * (frame_ms / 1000.0))
    if frame_len <= 0:
        return -1.0
    # Ensure multiple of frame_len
    usable = (len(segment) // frame_len) * frame_len
    if usable <= 0:
        return -1.0

    seg = segment[:usable]
    # Convert float32 [-1,1] to int16 bytes
    pcm16 = np.clip(seg * 32768.0, -32768, 32767).astype(np.int16).tobytes()

    num_frames = usable // frame_len
    speech = 0
    for i in range(num_frames):
        chunk = pcm16[i * frame_len * 2 : (i + 1) * frame_len * 2]
        try:
            if vad.is_speech(chunk, sr):
                speech += 1
        except Exception:
            # If VAD errors on a frame, skip it
            continue
    if num_frames == 0:
        return -1.0
    return float(speech) / float(num_frames)


def _speech_ratio_energy(segment: np.ndarray, sr: int) -> float:
    """Fallback speech ratio using energy gating over short frames."""
    frame_length = max(1, int(sr * 0.032))
    hop_length = max(1, int(sr * 0.008))
    rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return 0.0
    rms = np.nan_to_num(rms, nan=0.0)
    thr = max(0.1 * float(np.max(rms)), float(np.percentile(rms, 60)))
    speech_frames = float(np.count_nonzero(rms >= thr))
    total_frames = float(rms.size)
    return speech_frames / max(total_frames, 1.0)


def analyse_audio_features(
    cand: Candidate,
    audio_path: Path,
    *,
    sr: int = 16_000,
    margin: float = 0.4,
    energy_percentile: float = 40.0,
    max_trim: float = 0.6,
    vad_aggr: int = 2,
    bgm_threshold: float = 0.55,
    preloaded: Optional[Tuple[np.ndarray, int]] = None,
) -> Dict[str, float]:
    """Compute simple audio-based heuristics and refine boundaries for ``cand``."""

    if preloaded is not None:
        full, full_sr = preloaded
        use_sr = full_sr
        window_start = max(0.0, cand.start - margin)
        window_end = cand.end + margin
        start_idx = max(0, int(window_start * use_sr))
        end_idx = min(full.shape[-1], max(start_idx + 1, int(window_end * use_sr)))
        samples = full[start_idx:end_idx]
        sr = use_sr
    else:
        samples, sr, window_start = _load_audio_window(
            audio_path,
            start=cand.start,
            end=cand.end,
            sr=sr,
            margin=margin,
        )

    # 映射候选区间到窗口内的采样位置
    rel_start = max(cand.start - window_start, 0.0)
    rel_end = rel_start + cand.duration
    start_idx = int(rel_start * sr)
    end_idx = max(start_idx + 1, int(rel_end * sr))
    end_idx = min(end_idx, samples.shape[-1])

    segment = samples[start_idx:end_idx]
    if segment.size < sr // 10:  # 太短时直接退回原值
        return {}

    frame_length = int(sr * 0.032)
    hop_length = max(1, int(sr * 0.008))

    rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
    if not np.isfinite(rms).all():
        rms = np.nan_to_num(rms, nan=0.0)

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    coeff_var = float(rms_std / (rms_mean + 1e-8))

    # 频谱特征
    n_fft = int(2 ** np.ceil(np.log2(frame_length)))
    stft = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)) ** 2
    if stft.size == 0:
        stft = np.zeros((n_fft // 2 + 1, 1))
    flatness = float(librosa.feature.spectral_flatness(S=stft).mean())

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = (freqs >= 50) & (freqs <= 300)
    high_mask = freqs > 300
    low_energy = float(stft[low_mask].sum())
    high_energy = float(stft[high_mask].sum())
    low_ratio = float(low_energy / (low_energy + high_energy + 1e-8))

    # 估计背景音乐评分：谱平坦度越高、能量变化越小 -> 背景越严重
    flat_score = clamp((flatness - 0.1) / 0.9, 0.0, 1.0)
    var_score = clamp(0.4 - coeff_var, 0.0, 0.4) / 0.4
    band_score = clamp(0.4 - low_ratio, 0.0, 0.4) / 0.4
    bgm_score = clamp(0.6 * flat_score + 0.25 * var_score + 0.15 * band_score, 0.0, 1.0)

    # 语音占比 (WEbrtcVAD -> fallback)
    speech_ratio = _speech_ratio_webrtc(segment, sr, aggressiveness=vad_aggr)
    if speech_ratio < 0:
        speech_ratio = _speech_ratio_energy(segment, sr)

    # 简易 SNR 估计：以能量门限划分语音/非语音
    frame_length = int(sr * 0.032)
    hop_length = max(1, int(sr * 0.008))
    rms_frames = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
    rms_frames = np.nan_to_num(rms_frames, nan=0.0)
    if rms_frames.size > 0:
        thr_energy = max(0.1 * float(np.max(rms_frames)), float(np.percentile(rms_frames, 60)))
        speech_mask = rms_frames >= thr_energy
        non_speech_mask = ~speech_mask
        speech_energy = float(np.mean(rms_frames[speech_mask])) if np.any(speech_mask) else 1e-6
        noise_energy = float(np.mean(rms_frames[non_speech_mask])) if np.any(non_speech_mask) else 1e-6
        snr_db = 10.0 * np.log10((speech_energy + 1e-9) / (noise_energy + 1e-9))
    else:
        snr_db = 0.0

    # 静音边界微调
    energy = rms / (rms.max() + 1e-8)
    times = np.arange(len(energy)) * hop_length / sr
    threshold = max(0.1, np.percentile(energy, energy_percentile))

    voiced = np.where(energy >= threshold)[0]
    if voiced.size > 0:
        first_idx = int(voiced[0])
        last_idx = int(voiced[-1])
        trim_lead = min(times[first_idx], max_trim)
        trim_tail = min(times[-1] - times[last_idx], max_trim)
    else:
        trim_lead = 0.0
        trim_tail = 0.0

    # 静音评估仅用于记录，实际剪辑阶段统一补 1 秒静音
    lead_silence = float(trim_lead)
    trail_silence = float(trim_tail)
    silence_score = 1.0

    features = {
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "coeff_var": coeff_var,
        "spectral_flatness": flatness,
        "low_freq_ratio": low_ratio,
        "bgm_score": bgm_score,
        "speech_ratio": float(speech_ratio),
        "snr_db": float(snr_db),
        "lead_silence": lead_silence,
        "trail_silence": trail_silence,
        "silence_score": silence_score,
        "energy_threshold": threshold,
        "analysis_sr": sr,
    }

    # 根据阈值进行粗筛标记
    if bgm_score > bgm_threshold:
        cand.status = "noisy"
    # 不强制静音边界，保持原始时间段

    return features


def score_candidate(
    cand: Candidate,
    *,
    target_duration: float,
    audio_duration: Optional[float],
    skip_head: float,
    skip_tail: float,
    hard_skip_tail: float,
    min_duration: float,
    max_duration: float,
) -> None:
    """Attach a composite score to *cand* for ranking."""

    duration_term = clamp(1.0 - abs(cand.duration - target_duration) / target_duration, 0.0, 1.0)
    confidence_term = clamp(cand.avg_conf, 0.0, 1.0)
    density = cand.word_count / cand.duration if cand.duration > 0 else 0.0
    density_term = clamp((density - 1.5) / 2.5, 0.0, 1.0)

    penalties = 0.0
    position_term = 1.0
    if audio_duration:
        if cand.start < skip_head:
            penalties += 0.25
        if cand.end > audio_duration - skip_tail:
            penalties += 0.15
        midpoint = audio_duration / 2
        dist_mid = abs((cand.start + cand.end) / 2 - midpoint)
        position_term = 1.0 - clamp(dist_mid / midpoint, 0.0, 1.0) * 0.3
        # 硬性尾部丢弃
        if hard_skip_tail > 0 and cand.end > (audio_duration - hard_skip_tail):
            cand.status = "tail"

    if not (min_duration <= cand.duration <= max_duration):
        penalties += clamp(
            abs(cand.duration - clamp(cand.duration, min_duration, max_duration)) / max(min_duration, 1e-6),
            0.0,
            0.6,
        )
        cand.status = "out_of_range"

    features = getattr(cand, "audio_features", {})
    # 清洁度融合：背景音乐越低越好 + 语音占比越高越好
    bgm = float(features.get("bgm_score", 0.0))
    speech_ratio = float(features.get("speech_ratio", 1.0))
    speech_term = clamp((speech_ratio - 0.6) / 0.4, 0.0, 1.0)
    cleanliness_term = clamp(1.0 - bgm, 0.0, 1.0) * 0.6 + speech_term * 0.4
    silence_term = 1.0
    bgm_penalty = clamp(max(0.0, bgm - 0.35) * 1.5, 0.0, 0.7)
    speech_penalty = clamp(max(0.0, 0.7 - speech_ratio) / 0.7, 0.0, 1.0) * 0.4
    penalties += bgm_penalty
    penalties += speech_penalty

    total = (
        0.3 * duration_term
        + 0.25 * confidence_term
        + 0.15 * density_term
        + 0.15 * cleanliness_term
        + 0.15 * position_term
    ) - penalties
    cand.score_components = {
        "duration": duration_term,
        "confidence": confidence_term,
        "density": density_term,
        "position": position_term,
        "cleanliness": cleanliness_term,
        "silence": silence_term,
        "penalties": penalties,
        "total": total,
    }


def _overlap_seconds(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def pick_top(
    candidates: Dict[str, List[Candidate]],
    *,
    per_speaker: int,
    extra: int,
    min_duration: float,
    max_duration: float,
    allow_out_of_range: bool,
    min_speech_ratio: float,
    bgm_threshold: float,
    avoid_map: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    avoid_min_overlap: float = 0.5,
    sample_top_k: int = 0,
) -> Dict[str, Dict[str, List[Candidate]]]:
    """Select top-scoring candidates per speaker and keep extras for review."""

    grouped: Dict[str, Dict[str, List[Candidate]]] = {}
    for speaker, cand_list in candidates.items():
        avoid_list = (avoid_map or {}).get(speaker, [])
        eligible: List[Candidate] = []
        for c in cand_list:
            feats = getattr(c, "audio_features", {})
            speech_ratio = float(feats.get("speech_ratio", 1.0))
            bgm = float(feats.get("bgm_score", 0.0))
            if not (min_duration <= c.duration <= max_duration):
                continue
            if c.status in {"noisy", "tail"}:
                continue
            if speech_ratio < min_speech_ratio:
                continue
            if bgm > bgm_threshold:
                continue
            if avoid_list:
                if any(_overlap_seconds((c.start, c.end), (s, e)) >= avoid_min_overlap for s, e in avoid_list):
                    continue
            eligible.append(c)
        sorted_all = sorted(cand_list, key=lambda c: c.score, reverse=True)
        # 备选池：也需满足清洁度/语音占比/非尾部
        fallback_pool: List[Candidate] = []
        for c in cand_list:
            feats = getattr(c, "audio_features", {})
            speech_ratio = float(feats.get("speech_ratio", 1.0))
            bgm = float(feats.get("bgm_score", 0.0))
            if c.status in {"noisy", "tail"}:
                continue
            if speech_ratio < min_speech_ratio:
                continue
            if bgm > bgm_threshold:
                continue
            if avoid_list:
                if any(_overlap_seconds((c.start, c.end), (s, e)) >= avoid_min_overlap for s, e in avoid_list):
                    continue
            fallback_pool.append(c)
        # 始终按分数降序排序后再选择，避免出现“越往后越好”的错觉
        eligible_sorted = sorted(eligible, key=lambda c: c.score, reverse=True)
        fallback_sorted = sorted(fallback_pool, key=lambda c: c.score, reverse=True)
        pool = eligible_sorted if eligible_sorted else (fallback_sorted if allow_out_of_range else [])
        # 选择策略：支持 Top-K 内随机（在已排序的 pool 上）
        if sample_top_k and len(pool) > 0:
            top_k = pool[: min(sample_top_k, len(pool))]
            selected = random.sample(top_k, k=min(per_speaker, len(top_k)))
            # 若不足，继续按分数从剩余 pool 追加
            if len(selected) < per_speaker:
                for c in pool:
                    if c not in selected:
                        selected.append(c)
                        if len(selected) >= per_speaker:
                            break
        else:
            selected = pool[:per_speaker]
        # 备选也从备选池中挑，避免包含被丢弃的样本
        alt_pool_sorted = fallback_sorted
        alternatives = [c for c in alt_pool_sorted if c not in selected][:extra]
        grouped[speaker] = {
            "selected": selected,
            "alternatives": alternatives,
            "eligible_count": len(eligible),
        }
    return grouped


def confirm(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("请输入 y 或 n。")


def export_clip(
    *,
    audio: Path,
    output: Path,
    start: float,
    end: float,
    tail_silence: float,
    sample_rate: int,
    channels: int,
    codec: str,
    bitrate: str,
    afilter: Optional[str],
    dry_run: bool,
    verbose: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        if verbose:
            print(f"[dry-run] {output} -> {start:.3f}-{end:.3f}")
        return

    # 组装 -af 过滤器链
    if afilter and afilter.strip():
        af_expr = f"{afilter},apad=pad_dur={tail_silence}"
    else:
        af_expr = f"apad=pad_dur={tail_silence}"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error" if not verbose else "info",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(audio),
        "-af",
        af_expr,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        codec,
        "-b:a",
        bitrate,
        "-y",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"音频文件不存在: {args.audio}")
    if not args.metadata.is_file():
        raise SystemExit(f"元数据文件不存在: {args.metadata}")

    # 解析命令行显式提供的参数名，用于后续避免被 preset/config 覆盖
    def _explicit_flags(argv: List[str]) -> set[str]:
        names: set[str] = set()
        it = iter(argv)
        for tok in it:
            if not tok.startswith("--"):
                continue
            key = tok[2:].split("=", 1)[0]
            # 跳过带值参数的下一个 token（如果不是 -- 开头且当前 tok 不含=）
            try:
                peek = next(it)
                if peek.startswith("--") or "=" in tok:
                    # 不是值，推回（无法轻易推回，简单忽略）
                    pass
            except StopIteration:
                pass
            names.add(key.replace("-", "_"))
        return names

    explicit = _explicit_flags(sys.argv[1:])

    # 应用预设（仅覆盖未显式提供的参数）
    def _apply_if_missing(mapping: Dict[str, object]) -> None:
        for k, v in mapping.items():
            if k in explicit:
                continue
            if hasattr(args, k):
                setattr(args, k, v)

    if args.preset:
        if args.preset == "podcast":
            _apply_if_missing({
                "min_start": 420.0,
                "hard_skip_tail": 60.0,
                "per_speaker": 3,
                "min_speech_ratio": 0.75,
                "bgm_threshold": 0.50,
                "vad_aggressiveness": 2,
                "sample_top_k": 5,
                "workers": 8,
                "preload_audio": True,
                "afilter": "highpass=f=80,lowpass=f=8000,afftdn=nf=-28",
            })
        elif args.preset == "strict":
            _apply_if_missing({
                "min_start": 480.0,
                "hard_skip_tail": 90.0,
                "per_speaker": 2,
                "min_speech_ratio": 0.82,
                "bgm_threshold": 0.45,
                "vad_aggressiveness": 3,
                "sample_top_k": 5,
                "workers": 8,
                "preload_audio": True,
                "afilter": "highpass=f=100,lowpass=f=7500,afftdn=nf=-30",
            })
        elif args.preset == "relaxed":
            _apply_if_missing({
                "min_start": 300.0,
                "hard_skip_tail": 30.0,
                "per_speaker": 3,
                "min_speech_ratio": 0.65,
                "bgm_threshold": 0.6,
                "vad_aggressiveness": 1,
                "sample_top_k": 5,
                "workers": 4,
                "preload_audio": False,
                "afilter": "highpass=f=80,lowpass=f=9000",
            })
        elif args.preset == "fast":
            _apply_if_missing({
                "workers": 8,
                "preload_audio": True,
            })

    # 应用配置文件（亦仅覆盖未显式提供的参数）
    if args.config and args.config.is_file():
        try:
            data = json.loads(args.config.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _apply_if_missing({k: v for k, v in data.items() if hasattr(args, k)})
        except Exception as exc:
            print(f"[warn] 读取配置失败: {exc}")

    speakers = load_metadata(args.metadata)
    if not speakers:
        raise SystemExit("未在元数据中找到带说话人标签的词条")

    candidates = build_candidates(
        speakers,
        max_gap=args.max_gap,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_start=args.min_start,
    )
    audio_duration = ffprobe_duration(args.audio)

    # 并行/串行分析候选
    preloaded_audio: Optional[Tuple[np.ndarray, int]] = None
    if args.preload_audio:
        try:
            y_full, sr_full = librosa.load(args.audio, sr=16_000, mono=True)
            preloaded_audio = (np.nan_to_num(y_full, nan=0.0), sr_full)
        except Exception as exc:
            print(f"[warn] 预载音频失败，回退为分段载入: {exc}")
            preloaded_audio = None

    def _process(c: Candidate) -> Candidate:
        c.audio_features = analyse_audio_features(
            c,
            args.audio,
            sr=16_000,
            margin=0.4,
            vad_aggr=args.vad_aggressiveness,
            bgm_threshold=args.bgm_threshold,
            preloaded=preloaded_audio,
        )
        score_candidate(
            c,
            target_duration=args.target_duration,
            audio_duration=audio_duration,
            skip_head=args.skip_head,
            skip_tail=args.skip_tail,
            hard_skip_tail=args.hard_skip_tail,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        return c

    all_cands: List[Candidate] = [c for lst in candidates.values() for c in lst]
    if args.workers and args.workers > 0 and len(all_cands) > 1:
        with _futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_process, all_cands))
    else:
        for c in all_cands:
            _process(c)

    # 读取需要避开的历史片段
    def _load_avoid_intervals(paths: Optional[List[str]]) -> Dict[str, List[Tuple[float, float]]]:
        result: Dict[str, List[Tuple[float, float]]] = {}
        if not paths:
            return result
        for p in paths:
            path = Path(p)
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            margin = float(args.avoid_margin)
            for item in data.get("clips", []):
                if not isinstance(item, dict):
                    continue
                speaker = item.get("speaker")
                start = item.get("start")
                end = item.get("end")
                if not speaker or start is None or end is None:
                    continue
                s = max(0.0, float(start) - margin)
                e = float(end) + margin
                result.setdefault(speaker, []).append((s, e))
        return result

    avoid_map = _load_avoid_intervals(args.avoid_manifest)
    if args.random_seed is not None:
        random.seed(args.random_seed)

    grouped = pick_top(
        candidates,
        per_speaker=args.per_speaker,
        extra=args.extra_candidates,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        allow_out_of_range=args.allow_out_of_range,
        min_speech_ratio=args.min_speech_ratio,
        bgm_threshold=args.bgm_threshold,
        avoid_map=avoid_map,
        avoid_min_overlap=args.avoid_min_overlap,
        sample_top_k=args.sample_top_k,
    )

    def _has_any_selection(g: Dict[str, Dict[str, List[Candidate]]]) -> bool:
        for _, bundle in g.items():
            if bundle.get("selected"):
                return True
        return False

    def _detect_repo_root() -> Path:
        """Best-effort detect project root for Codex -C; fallback to CWD."""
        here = Path.cwd()
        try:
            result = _subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            )
            root = Path(result.stdout.strip())
            if root.is_dir():
                return root
        except Exception:
            pass
        # fallback to repo root heuristic: two-level up from this file
        return Path(__file__).resolve().parents[1]

    def _build_codex_prompt(ctx: Dict[str, object]) -> str:
        ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
        guide = (
            "你是语音样本提取助手。给定 WhisperX 分段与候选统计，\n"
            "请仅返回一个 JSON，键为 params，值为建议的参数集合。\n\n"
            "目标：\n"
            "- 为每个说话人选出高质量、干净、时长合适(默认6-12s，目标8s)的语音片段；\n"
            "- 提高语音占比(speech_ratio)、降低背景音乐(bgm_score)，避开片头/尾；\n"
            "- 不要给出解释性文本，不要输出除 JSON 外的任何内容。\n\n"
            "可调整的参数键（按需提供）：\n"
            "min_start, skip_head, hard_skip_tail, min_duration, max_duration, target_duration, max_gap, \n"
            "per_speaker, min_speech_ratio, bgm_threshold, vad_aggressiveness, allow_out_of_range, sample_top_k, afilter\n\n"
            "若需要启用降噪/均衡，可建议 afilter，例如：\n"
            "'highpass=f=80,lowpass=f=8000,afftdn=nf=-28' 或 'highpass=f=100,lowpass=f=7500'。\n\n"
            "输入(JSON)：\n"
        )
        tail = (
            "\n\n输出(JSON 仅此一行)：\n"
            "{\n  \"params\": { ... }\n}"
        )
        return f"{guide}{ctx_json}{tail}"

    def _assistant_suggest() -> bool:
        """Call assistant (codex/external) to suggest new params; return True if applied and improved."""
        context: Dict[str, object] = {
            "audio": str(args.audio),
            "metadata": str(args.metadata),
            "audio_duration": audio_duration,
            "speakers": sorted(list(speakers.keys())),
            "params": {
                "min_start": args.min_start,
                "skip_head": args.skip_head,
                "hard_skip_tail": args.hard_skip_tail,
                "min_duration": args.min_duration,
                "max_duration": args.max_duration,
                "target_duration": args.target_duration,
                "max_gap": args.max_gap,
                "per_speaker": args.per_speaker,
                "min_speech_ratio": args.min_speech_ratio,
                "bgm_threshold": args.bgm_threshold,
                "vad_aggressiveness": args.vad_aggressiveness,
                "allow_out_of_range": args.allow_out_of_range,
                "sample_top_k": args.sample_top_k,
            },
            "candidates_per_speaker": {k: len(v) for k, v in candidates.items()},
        }
        if args.assistant_log:
            try:
                log = {"request": context}
                args.assistant_log.parent.mkdir(parents=True, exist_ok=True)
                args.assistant_log.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        reply: Optional[dict] = None
        if args.assistant_engine == "codex":
            print("[assistant] 使用 Codex CLI 生成参数建议…")
            # 组装 codex exec 命令
            flags = shlex.split(args.assistant_codex_flags or "")
            workdir = args.assistant_codex_workdir or _detect_repo_root()
            cmd: List[str] = ["codex", "exec", *flags, "-C", str(workdir)]
            if args.assistant_codex_model:
                cmd.extend(["-m", str(args.assistant_codex_model)])
            # 从 stdin 读取 prompt
            # 将最后一条回复写入临时文件，避免解析额外日志
            with tempfile.NamedTemporaryFile(prefix="codex_assistant_", suffix=".json", delete=False) as tf:
                tmp_path = Path(tf.name)
            cmd.extend(["-", "--output-last-message", str(tmp_path)])
            prompt = _build_codex_prompt(context)
            try:
                proc = _subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=max(10, int(args.assistant_timeout)),
                )
            except Exception as exc:
                print(f"[assistant] Codex 执行失败: {exc}")
                return False
            if proc.returncode != 0:
                print(f"[assistant] Codex 返回非零码: {proc.returncode}\n{proc.stderr}")
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                return False
            try:
                raw = tmp_path.read_text(encoding="utf-8")
                reply = json.loads(raw)
            except Exception as exc:
                print(f"[assistant] 无法解析 Codex 最终输出为 JSON: {exc}")
                return False
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
        elif args.assistant_engine == "external" or args.assistant_cmd:
            # 兼容外部助手命令：从 stdin 接收 JSON 上下文，stdout 输出 JSON
            print("[assistant] 调用外部助手以获取参数建议…")
            req = json.dumps(context, ensure_ascii=False)
            try:
                proc = _subprocess.run(
                    args.assistant_cmd,
                    input=req,
                    text=True,
                    capture_output=True,
                    timeout=max(10, int(args.assistant_timeout)),
                    shell=True,
                )
            except Exception as exc:
                print(f"[assistant] 外部助手执行失败: {exc}")
                return False
            if proc.returncode != 0:
                print(f"[assistant] 外部助手返回非零码: {proc.returncode}\n{proc.stderr}")
                return False
            try:
                reply = json.loads(proc.stdout)
            except Exception as exc:
                print(f"[assistant] 无法解析助手输出为 JSON: {exc}")
                return False
        else:
            # 未开启助手
            return False

        if args.assistant_log and reply is not None:
            try:
                log = json.loads(args.assistant_log.read_text(encoding="utf-8"))
                log["response"] = reply
                args.assistant_log.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        if reply is None:
            return False
        params = reply.get("params") if isinstance(reply, dict) else None
        if not isinstance(params, dict):
            print("[assistant] 响应中未找到 params dict，忽略。")
            return False
        # 应用建议参数
        def _maybe(name: str, cast):
            if name in params:
                try:
                    setattr(args, name, cast(params[name]))
                    print(f"[assistant] 应用建议: {name} -> {getattr(args, name)}")
                except Exception:
                    print(f"[assistant] 忽略无效建议: {name}={params.get(name)!r}")
        _maybe("min_start", float)
        _maybe("skip_head", float)
        _maybe("hard_skip_tail", float)
        _maybe("min_duration", float)
        _maybe("max_duration", float)
        _maybe("target_duration", float)
        _maybe("max_gap", float)
        _maybe("per_speaker", int)
        _maybe("min_speech_ratio", float)
        _maybe("bgm_threshold", float)
        _maybe("vad_aggressiveness", int)
        if "allow_out_of_range" in params:
            setattr(args, "allow_out_of_range", bool(params["allow_out_of_range"]))
            print(f"[assistant] 应用建议: allow_out_of_range -> {args.allow_out_of_range}")
        _maybe("sample_top_k", int)

        # 重新构建候选与打分（保留已计算的 audio_features，更新打分与筛选阈值）
        for c in all_cands:
            score_candidate(
                c,
                target_duration=args.target_duration,
                audio_duration=audio_duration,
                skip_head=args.skip_head,
                skip_tail=args.skip_tail,
                hard_skip_tail=args.hard_skip_tail,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
        nonlocal_grouped = pick_top(
            candidates,
            per_speaker=args.per_speaker,
            extra=args.extra_candidates,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            allow_out_of_range=args.allow_out_of_range,
            min_speech_ratio=args.min_speech_ratio,
            bgm_threshold=args.bgm_threshold,
            avoid_map=avoid_map,
            avoid_min_overlap=args.avoid_min_overlap,
            sample_top_k=args.sample_top_k,
        )
        if _has_any_selection(nonlocal_grouped):
            nonlocal grouped
            grouped = nonlocal_grouped
            print("[assistant] 新参数生效，已选出样本。")
            return True
        print("[assistant] 应用建议后仍未选出样本。")
        return False

    if not _has_any_selection(grouped):
        _assistant_suggest()

    manifest_clips: List[dict] = []
    # 供下游快速消费的摘要：每个说话人的已导出样本文件与备选
    summary_selected: Dict[str, List[str]] = {}
    summary_alternatives: Dict[str, List[dict]] = {}
    speaker_order = sorted(grouped.keys())
    speaker_index = {speaker: idx + 1 for idx, speaker in enumerate(speaker_order)}

    for speaker in speaker_order:
        bundle = grouped[speaker]
        selected = bundle["selected"]
        if not selected:
            print(f"[warn] {speaker} 没有满足条件的片段")
            continue

        for idx, cand in enumerate(selected, 1):
            if cand.status == "tail":
                print(
                    f"[warn] 跳过 {speaker} 的候选 {cand.start:.3f}-{cand.end:.3f}s (命中最后 {args.hard_skip_tail:.0f}s 丢弃规则)"
                )
                continue
            if cand.status == "out_of_range" and not args.allow_out_of_range:
                print(
                    f"[warn] 跳过 {speaker} 的候选 {cand.start:.3f}-{cand.end:.3f}s (不在时长范围)"
                )
                continue

            # 文件命名：尽量复用 JSON 标号（如 SPEAKER_00 -> speaker0），否则回退到顺序索引
            def _suffix_number(label: str) -> Optional[int]:
                m = re.search(r"(?:^|[_-])(\d+)$", label)
                return int(m.group(1)) if m else None

            suffix = _suffix_number(speaker)
            file_idx = suffix if suffix is not None else speaker_index[speaker]
            speaker_label = f"speaker{file_idx}"
            filename = f"{speaker_label}_{idx}.mp3" if args.per_speaker > 1 else f"{speaker_label}.mp3"
            output_path = args.output_dir / filename

            if args.interactive:
                print("-" * 60)
                print(f"说话人 {speaker} ({speaker_label}) 候选 {idx}")
                print(f"时间: {cand.start:.3f}-{cand.end:.3f} ({cand.duration:.2f}s)")
                print(f"文本: {cand.text}")
                print(f"得分: {cand.score:.3f} -> {cand.score_components}")
                if not confirm("导出这个片段吗?"):
                    continue

            # 导出前对边界进行轻微内缩
            lead_cut = min(float(cand.audio_features.get("lead_silence", 0.0) or 0.0), float(args.edge_trim_max))
            trail_cut = min(float(cand.audio_features.get("trail_silence", 0.0) or 0.0), float(args.edge_trim_max))
            new_start = cand.start + lead_cut
            new_end = cand.end - trail_cut
            if new_end - new_start < 0.5:
                new_start, new_end = cand.start, cand.end

            export_clip(
                audio=args.audio,
                output=output_path,
                start=new_start,
                end=new_end,
                tail_silence=args.tail_silence,
                sample_rate=args.sample_rate,
                channels=args.channels,
                codec=args.codec,
                bitrate=args.bitrate,
                afilter=args.afilter,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )

            manifest_clips.append(
                {
                    "speaker": speaker,
                    "speaker_index": speaker_index[speaker],
                    "file": str(output_path),
                    "text": cand.text,
                    "start": cand.start,
                    "end": cand.end,
                    "exported_start": new_start,
                    "exported_end": new_end,
                    "duration": cand.duration,
                    "score": cand.score,
                    "score_components": cand.score_components,
                    "status": cand.status,
                    "audio_features": cand.audio_features,
                }
            )
            # 记录摘要（多数场景每说话人只导出一个样本，仍支持多样本）
            norm_label = speaker_label
            summary_selected.setdefault(norm_label, []).append(str(output_path))

        alternatives = [
            {
                "speaker": speaker,
                "speaker_index": speaker_index[speaker],
                "text": cand.text,
                "start": cand.start,
                "end": cand.end,
                "duration": cand.duration,
                "score": cand.score,
                "score_components": cand.score_components,
                "status": cand.status,
                "audio_features": cand.audio_features,
            }
            for cand in bundle["alternatives"]
        ]
        if alternatives:
            manifest_clips.append({"speaker": speaker, "alternatives": alternatives})
            # 将备选纳入摘要，便于后续可视化或再生成
            summary_alternatives[norm_label] = alternatives

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "audio": str(args.audio),
                    "metadata": str(args.metadata),
                    "output_dir": str(args.output_dir),
                    "clips": manifest_clips,
                    "summary": {
                        "selected": summary_selected,
                        "alternatives": summary_alternatives,
                        "speakers": sorted(summary_selected.keys()),
                    },
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    print("生成完成。")
    print(f"输出目录: {args.output_dir}")
    if args.manifest:
        print(f"清单文件: {args.manifest}")


if __name__ == "__main__":
    main()
