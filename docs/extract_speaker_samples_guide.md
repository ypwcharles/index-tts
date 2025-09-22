# Extract Speaker Samples — 使用文档

- 工具路径: `tools/extract_speaker_samples.py`
- 功能: 从 WhisperX 的分段与说话人标注结果中，为每位说话人导出若干条干净、可作克隆/建模使用的音频样本，并生成清单用于复现、避让与审阅。

## 依赖与环境
- 必需: `ffmpeg/ffprobe`、Python 3.10+、`librosa`（已在项目依赖中）
- 可选: `webrtcvad`（更稳健的 VAD）
- 安装示例:
  - 项目依赖: `uv sync`
  - 可选 VAD: `.venv/bin/python -m pip install webrtcvad`

## 输入与输出
- 输入
  - 源音频: `*.mp3|*.wav|*.m4a|*.flac`
  - WhisperX JSON: 包含 `segments[*].words[*]`，词级别含 `start/end/speaker`
- 输出
  - 音频样本（默认 MP3），命名与 JSON 说话人编号对齐:
    - 单样本: `speakerN.mp3`（如 `SPEAKER_00` → `speaker0`；无法解析时回退顺序编号 1、2…）
    - 多样本: `speakerN_i.mp3`（i 从 1 开始，分数从高到低）
  - 清单 JSON（`--manifest`）
    - 已导出样本条目: `speaker/speaker_index/file/text/start/end/exported_start/exported_end/duration/score/...`
    - 备选集合条目: `{"speaker": "...", "alternatives": [ {...}, ... ]}`（不含 file 字段）

## 快速开始（推荐用预设）
- 常用播客场景（干净、人声优先、避开片头片尾）
  - `.venv/bin/python tools/extract_speaker_samples.py <audio> <whisperx.json> --output-dir <out> --manifest <out>/manifest.json --preset podcast`
- 更严格（更干净、样本更少）
  - `--preset strict`
- 偏速度（更快产出）
  - `--preset fast`

说明
- 预设会一键设置你常用的参数组合：如 `min_start/hard_skip_tail/per_speaker/min_speech_ratio/bgm_threshold/vad_aggressiveness/workers/preload_audio/afilter` 等。
- 显式传入的参数优先级更高，不会被预设覆盖。

## 常见用法与技巧
- 每人导出 3 段，并避开上一版清单（重生成不踩旧片段）
  - `.venv/bin/python tools/extract_speaker_samples.py <audio> <meta.json> --output-dir <out_v2> --manifest <out_v2>/manifest.json --preset podcast --avoid-manifest <out_v1>/manifest.json`
- “换一批试试”（Top‑K 随机抽样）
  - 加 `--sample-top-k 5 --random-seed 42`（Top‑K 越大，多样性越高，可重复抽样）
- 更严格避广告/口播（起点更晚 + 更强语音占比 + 更低 BGM）
  - `--min-start 480 --min-speech-ratio 0.8 --bgm-threshold 0.45`

## 关键参数（精简）
- 片头片尾
  - `--min-start` 跳过起始 N 秒（默认 300）
  - `--hard-skip-tail` 丢弃最后 N 秒（默认 60）
  - `--skip-head/--skip-tail` 仅降权靠近头/尾的片段（默认 300/180）
- 质量过滤
  - `--min-speech-ratio` 最小语音占比（默认 0.75）
  - `--vad-aggressiveness` VAD 严格度 0–3（默认 2）
  - `--bgm-threshold` 背景音乐评分阈值（默认 0.55；更低更严格）
- 时长/拼接
  - `--min-duration/--max-duration/--target-duration`（默认 6/12/8 秒）
  - `--max-gap` 合并词的最大静音间隔（默认 0.7s）
  - `--edge-trim-max` 导出前首尾静音的最大内缩（默认 0.2s）
  - `--tail-silence` 导出末尾补静音秒数（默认 1.0s）
- 选择策略
  - `--per-speaker N` 每位人数（默认 1）
  - `--extra-candidates K` 每人保留 K 个备选到清单
  - `--sample-top-k` 在得分 Top‑K 内随机抽（配合 `--random-seed`）
  - `--avoid-manifest` 避开历史清单；`--avoid-margin`（默认 1.5s 扩边）、`--avoid-min-overlap`（默认 0.5s）
- 性能
  - `--workers N` 并行线程（默认 0=串行；推荐 4–8）
  - `--preload-audio` 预载整段音频到 16k 单声道（更快但吃内存）
- 导出
  - `--afilter` 自定义 `ffmpeg -af`（会在末尾自动追加 `apad=pad_dur=...`）
  - `--sample-rate/--channels/--codec/--bitrate` 输出参数
  - `--dry-run` 只打分不导出；`--interactive` 逐条确认

## 配置文件与预设
- 预设: `--preset podcast|strict|relaxed|fast`
- 配置文件: `--config my.json`（内容如 `{ "min_start": 420, "per_speaker": 3, ... }`）
- 优先级: 显式参数 > 配置文件 > 预设

## 音质与性能建议
- 音质（通用人声清洁）: `--afilter "highpass=f=80,lowpass=f=8000,afftdn=nf=-28"`
- 更严格清洁: `afftdn` 更强，或叠加 `deesser`（谨慎）
- 性能: 长音频/候选多 → `--workers 8 --preload-audio`

## 常见问题
- 没有导出任何样本 → 放宽过滤：提高 `--bgm-threshold`（如 0.6）、降低 `--min-speech-ratio`（如 0.7），或关闭 `--hard-skip-tail`（设 0）
- 仍采到广告口播 → 提高 `--min-start`（如 480–600），并叠加 `--avoid-manifest` 避开上一批
- 剪切边界不够“干净” → 提高 `--edge-trim-max`（如 0.3–0.4），或加大 `--tail-silence`（如 1.5）

## 示例配置（podcast 预设风格）
参见: `docs/extract_speaker_samples.podcast.json`

## 变更要点（相对基础版本）
- VAD（webrtcvad）+ 能量回退，计算语音占比并用于过滤与打分
- 背景音乐评分（谱平坦度/能量波动/频带比）降权嘈杂片段
- 强制跳过片头/片尾；可设置“最后 N 秒直接丢弃”
- 切分优先选择更大的词间停顿，减少切在半句中间
- 导出前轻微内缩首尾静音，边界更干净
- 支持避让历史清单，重生成时不踩旧片段；支持 Top‑K 抽样“换一批”
- 并行分析与可选整段预载，显著提升性能

