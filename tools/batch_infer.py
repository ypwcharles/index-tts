import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as tomllib  # type: ignore

from indextts.infer_v2 import IndexTTS2


@dataclass
class VoiceConfig:
    name: str
    ref_audio: str
    emo_audio: Optional[str] = None
    emo_alpha: float = 1.0
    emo_text: Optional[str] = None
    emo_vector: Optional[Sequence[float]] = None


@dataclass
class RuntimeConfig:
    cfg_path: str
    model_dir: str
    output_dir: str
    output_file: str
    subtitle_path: Optional[str]
    remove_silence: bool
    interval_silence_ms: int
    max_text_tokens_per_segment: int
    device: Optional[str]
    use_fp16: bool
    use_cuda_kernel: Optional[bool]
    use_deepspeed: bool
    verbose: bool


def load_toml(path: str) -> Dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def resolve_text(config: Dict, text_override: Optional[str], file_override: Optional[str]) -> str:
    if text_override:
        return text_override
    if file_override:
        return read_text_file(file_override)

    inline = config.get("gen_text", "").strip()
    if inline:
        return inline

    story_file = config.get("gen_file", "").strip()
    if not story_file:
        raise ValueError("在配置中未找到 gen_text 或 gen_file，且未通过命令行指定文本。")
    return read_text_file(story_file)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def normalize_story_script(script: str) -> str:
    """Normalize raw script so that each [speaker] section starts a new line.

    This enables inputs where multiple [speaker] tags appear on the same line
    (e.g. "[speaker0]: hi. [speaker1]: hello.") to be parsed correctly by
    the line-based parser below.

    The strategy is to scan for occurrences of a bracketed tag like
    "[speakerX]" (optionally followed by colon/space) and split the text into
    segments, each beginning with such a tag. Any text between two tags is
    kept as the content for the preceding tag. Content before the first tag is
    ignored.
    """
    # Normalize newlines and trim outer whitespace
    s = script.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Only split at speaker tags like [speaker0], [speakerA], ... (case-insensitive).
    tag_iter = list(re.finditer(r"\[\s*speaker[^\]]*\]", s, flags=re.IGNORECASE))
    if not tag_iter:
        return script
    parts: List[str] = []
    for idx, m in enumerate(tag_iter):
        start = m.start()
        next_start = tag_iter[idx + 1].start() if idx + 1 < len(tag_iter) else len(s)
        segment = s[start:next_start].strip()
        if segment:
            parts.append(segment)
    return "\n".join(parts)


def parse_story(script: str) -> List[Tuple[str, str]]:
    # Preprocess so multiple [speaker] tokens on one line get split by lines
    script = normalize_story_script(script)
    pattern = re.compile(r"^\s*\[(?P<speaker>[^\]]+)\]\s*(?P<text>.*\S)?\s*$")
    result: List[Tuple[str, str]] = []
    for idx, raw in enumerate(script.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = pattern.match(raw)
        if not match:
            raise ValueError(f"第 {idx} 行缺少说话人标记: {raw}")
        speaker = match.group("speaker").strip()
        text = match.group("text") or ""
        text = text.strip()
        if not text:
            continue
        if result and result[-1][0] == speaker:
            result[-1] = (speaker, result[-1][1] + " " + text)
        else:
            result.append((speaker, text))
    if not result:
        raise ValueError("未解析到任何带说话人标记的文本。")
    return result


def load_voices(config: Dict) -> Dict[str, VoiceConfig]:
    voices_cfg = config.get("voices") or {}
    voices: Dict[str, VoiceConfig] = {}
    for name, raw in voices_cfg.items():
        ref_audio = raw.get("ref_audio")
        if not ref_audio:
            raise ValueError(f"说话人 {name} 缺少 ref_audio 字段。")
        emo_alpha = float(raw.get("emo_alpha", 1.0))
        emo_audio = raw.get("emo_audio") or None
        emo_text = raw.get("emo_text") or None
        emo_vector = raw.get("emo_vector")
        if emo_vector is not None and not isinstance(emo_vector, (list, tuple)):
            raise ValueError(f"说话人 {name} 的 emo_vector 必须是数组。")
        voices[name] = VoiceConfig(
            name=name,
            ref_audio=ref_audio,
            emo_audio=emo_audio,
            emo_alpha=emo_alpha,
            emo_text=emo_text,
            emo_vector=emo_vector,
        )
    if not voices:
        raise ValueError("配置文件中未定义任何 voices。")
    return voices


def build_runtime(config: Dict, args: argparse.Namespace) -> RuntimeConfig:
    runtime_cfg = config.get("runtime", {})
    output_dir = args.output_dir or config.get("output_dir") or "outputs"
    output_file = args.output_file or config.get("output_file") or "generated.wav"
    subtitle_file = args.subtitle_file or config.get("output_subtitle_file")

    return RuntimeConfig(
        cfg_path=runtime_cfg.get("cfg_path", "checkpoints/config.yaml"),
        model_dir=runtime_cfg.get("model_dir", "checkpoints"),
        output_dir=output_dir,
        output_file=output_file,
        subtitle_path=subtitle_file,
        remove_silence=bool(config.get("remove_silence", False)),
        interval_silence_ms=int(runtime_cfg.get("interval_silence", 200)),
        max_text_tokens_per_segment=int(runtime_cfg.get("max_text_tokens_per_segment", 120)),
        device=args.device or runtime_cfg.get("device"),
        use_fp16=bool(runtime_cfg.get("use_fp16", False)) if args.use_fp16 is None else args.use_fp16,
        use_cuda_kernel=runtime_cfg.get("use_cuda_kernel") if args.use_cuda_kernel is None else args.use_cuda_kernel,
        use_deepspeed=bool(runtime_cfg.get("use_deepspeed", False)) if args.use_deepspeed is None else args.use_deepspeed,
        verbose=args.verbose,
    )


def instantiate_tts(runtime: RuntimeConfig) -> IndexTTS2:
    return IndexTTS2(
        cfg_path=runtime.cfg_path,
        model_dir=runtime.model_dir,
        device=runtime.device,
        use_fp16=runtime.use_fp16,
        use_cuda_kernel=runtime.use_cuda_kernel,
        use_deepspeed=runtime.use_deepspeed,
    )


def synthesize_segment(
    tts: IndexTTS2,
    voice: VoiceConfig,
    text: str,
    runtime: RuntimeConfig,
) -> Tuple[int, torch.Tensor]:
    infer_kwargs = dict(
        emo_alpha=voice.emo_alpha,
        interval_silence=0,
        max_text_tokens_per_segment=runtime.max_text_tokens_per_segment,
        verbose=runtime.verbose,
    )
    if voice.emo_audio:
        infer_kwargs["emo_audio_prompt"] = voice.emo_audio
    if voice.emo_vector is not None:
        infer_kwargs["emo_vector"] = list(voice.emo_vector)
    if voice.emo_text:
        infer_kwargs["use_emo_text"] = True
        infer_kwargs["emo_text"] = voice.emo_text

    result = tts.infer(
        spk_audio_prompt=voice.ref_audio,
        text=text,
        output_path=None,
        **infer_kwargs,
    )

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("推理返回结果异常，未获得采样率与音频数据。")
    sampling_rate, wav_np = result
    wav_tensor = torch.from_numpy(wav_np.T).to(torch.float32) / 32767.0
    return sampling_rate, wav_tensor


def insert_silence(channel: int, samples: int) -> torch.Tensor:
    if samples <= 0:
        return torch.zeros(channel, 0)
    return torch.zeros(channel, samples)


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_audio(path: str, audio: torch.Tensor, sampling_rate: int) -> None:
    audio = audio.clamp(-1.0, 1.0).to(torch.float32)
    torchaudio.save(path, audio, sampling_rate)


def maybe_trim_silence(audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("需要 librosa 才能执行静音裁剪，请先安装该依赖。") from exc

    channel, _ = audio.shape
    flattened = audio.squeeze(0).cpu().numpy()
    if channel > 1:
        flattened = flattened.mean(axis=0)
    trimmed, index = librosa.effects.trim(flattened, top_db=20)
    if trimmed.size == 0:
        return audio
    start, end = index
    return audio[:, start:end]


def build_subtitle_entry(start: float, end: float, speaker: str, text: str) -> Dict:
    return {
        "speaker": speaker,
        "text": text,
        "start": round(start, 3),
        "end": round(end, 3),
    }


def dump_subtitles(path: str, entries: List[Dict]) -> None:
    payload = {"subtitles": entries}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量多说话人推理脚本")
    parser.add_argument("--config", required=True, help="story.toml 配置文件路径")
    parser.add_argument("--text", help="覆盖配置中的文本内容")
    parser.add_argument("--text-file", dest="text_file", help="覆盖配置中的文本文件路径")
    parser.add_argument("--output-dir", help="覆盖输出目录")
    parser.add_argument("--output-file", help="覆盖输出文件名")
    parser.add_argument("--subtitle-file", help="覆盖字幕文件名")
    parser.add_argument("--device", help="强制指定推理设备，如 cuda:0")
    parser.add_argument("--use-fp16", dest="use_fp16", action="store_true", help="开启 FP16 推理")
    parser.add_argument("--no-use-fp16", dest="use_fp16", action="store_false", help="关闭 FP16 推理")
    parser.set_defaults(use_fp16=None)
    parser.add_argument("--use-cuda-kernel", dest="use_cuda_kernel", action="store_true", help="启用 BigVGAN CUDA kernel")
    parser.add_argument("--no-use-cuda-kernel", dest="use_cuda_kernel", action="store_false", help="禁用 BigVGAN CUDA kernel")
    parser.set_defaults(use_cuda_kernel=None)
    parser.add_argument("--use-deepspeed", dest="use_deepspeed", action="store_true", help="开启 DeepSpeed")
    parser.add_argument("--no-use-deepspeed", dest="use_deepspeed", action="store_false", help="关闭 DeepSpeed")
    parser.set_defaults(use_deepspeed=None)
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_toml(args.config)
    voices = load_voices(config)
    runtime = build_runtime(config, args)

    text = resolve_text(config, args.text, args.text_file)
    story = parse_story(text)

    tts = instantiate_tts(runtime)

    generated: List[torch.Tensor] = []
    subtitles: List[Dict] = []
    sampling_rate: Optional[int] = None
    channel_count: Optional[int] = None
    current_time = 0.0
    interval_samples = 0

    total_segments = len(story)
    print(f"共 {total_segments} 个片段待合成。")

    for idx, (speaker, sentence) in enumerate(story, start=1):
        voice = voices.get(speaker)
        if not voice:
            raise KeyError(f"文本中出现未知说话人 {speaker}")

        preview = sentence.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"[{idx}/{total_segments}] {speaker}: {preview}")

        sr, audio = synthesize_segment(tts, voice, sentence, runtime)
        if sampling_rate is None:
            sampling_rate = sr
            channel_count = audio.shape[0]
            interval_samples = int(sampling_rate * runtime.interval_silence_ms / 1000.0)
        elif sr != sampling_rate:
            raise RuntimeError("不同片段的采样率不一致，推理结果异常。")

        duration = audio.shape[1] / sampling_rate
        start_time = current_time
        end_time = start_time + duration
        subtitles.append(build_subtitle_entry(start_time, end_time, speaker, sentence))
        generated.append(audio)
        current_time = end_time

        if idx != len(story) and interval_samples > 0:
            generated.append(insert_silence(channel_count, interval_samples))
            current_time += interval_samples / sampling_rate

    if sampling_rate is None:
        raise RuntimeError("未成功生成任何音频片段。")

    final_audio = torch.cat(generated, dim=1)

    if runtime.remove_silence:
        final_audio = maybe_trim_silence(final_audio, sampling_rate)

    ensure_dir(runtime.output_dir)
    output_path = os.path.join(runtime.output_dir, runtime.output_file)
    save_audio(output_path, final_audio, sampling_rate)
    print(f"音频已保存: {output_path}")

    if runtime.subtitle_path:
        subtitle_path = runtime.subtitle_path
        if not os.path.isabs(subtitle_path):
            subtitle_path = os.path.join(runtime.output_dir, subtitle_path)
        ensure_dir(os.path.dirname(subtitle_path) or ".")
        dump_subtitles(subtitle_path, subtitles)
        print(f"字幕已保存: {subtitle_path}")

    total_seconds = final_audio.shape[-1] / sampling_rate
    print(f"总时长: {total_seconds:.2f} 秒，片段数量: {len(story)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
