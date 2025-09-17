# 远程语音合成流程说明

## 改动概览

- 新增 `tools/batch_infer.py`：支持从 `story.toml` 与 `[speaker]` 文本批量合成，输出片段进度、字幕与音频。
- 新增并迭代 `tools/remote_runner.py`：一键完成上传、远程执行、日志采集与结果回传；若目录中缺少 `story.toml` 会自动生成默认配置。
- 远程脚本自动识别本地目录中的 `story.toml`、文本与多个说话人音频；自动推断远程工作目录、仓库路径。
- 推理阶段输出每个语句的进度提示，并生成包含总时长、耗时、RTF 等指标的日志文件。
- 完成后自动下载音频、字幕、日志到源目录，保持同步。

## 前置准备

1. 本地仓库根目录安装依赖：
   ```bash
   uv pip install tomli paramiko
   ```
2. 确保远端机器已拉取最新代码 (`git pull`) 并具备 `uv`、模型权重、CUDA 等环境。

## 本地自动化脚本使用

1. 在终端执行：
   ```bash
   PYTHONPATH="$PWD" uv run python tools/remote_runner.py
   ```
2. 按提示操作：
- 输入素材目录——脚本会自动生成 `story.toml`（若不存在），并仅使用目录中 `speakerN.*` 音频与同名 `<目录名>.txt` 文本。
   - 输入 SSH 命令（例如 `ssh -p 16688 root@connect.westc.gpuhub.com`）与密码。
   - 确认 FP16、BigVGAN、DeepSpeed 等开关（直接回车沿用 `story.toml` 中的默认设置，例如 `use_fp16=true`、`use_cuda_kernel=true`、`use_deepspeed=false`）。
3. 脚本流程：
   - 自动上传所需文件并在远端生成带时间戳的 story 配置。
   - 在 `/root/index-tts` 内执行批量合成，并将控制台输出写入日志（`story-<timestamp>.log`）。
   - 下载音频、字幕、日志到本地源目录（同时保留一份日志在远端工作目录）。
4. 推理完成后，生成的音频、字幕与日志会直接写回该目录，无需手动同步：
   - `<story>.mp3`
   - `subtitle-<story>.json`
   - `<story>-<timestamp>.log`
   - `<story>-<timestamp>-summary.txt`（包含 wall clock 耗时与总体 RTF 汇总）

## 远端手动执行示例

如需手动在远端执行，可参考：
```bash
cd /root/index-tts
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"  # 若需要镜像
PYTHONPATH="$PWD" uv run python tools/batch_infer.py \
  --config /root/autodl-fs/cc/story.toml \
  --text-file /root/autodl-fs/cc/cc.txt \
  --output-dir /root/autodl-fs/cc \
  --output-file cc.mp3 \
  --subtitle-file subtitle-cc.json \
  --device cuda:0 --use-fp16 --use-cuda-kernel --no-use-deepspeed
```

## 日志与进度信息

- `batch_infer.py` 会输出“`共 N 个片段`”、“`[idx/total] speaker: 摘要`”等信息，便于监控。
- 生成日志包含：
  - `Generated audio length`
  - `Total inference time`
  - `RTF`
- 日志文件会保存在本地源目录与远端工作目录中，命名为 `<story>-<timestamp>.log`。

## 常见问题

- **缺少依赖**：若脚本提示 `tomllib` 或 `paramiko` 缺失，确认是否已在 `.venv` 中执行 `uv pip install tomli paramiko`。
- **远程 `uv` 未找到**：脚本会自动导入 `/root/miniconda3/bin` 到 `PATH`，若仍失败，请确认远端安装位置是否正确。
- **网络超时**：可在 `story.toml` 中加入 `hf_endpoint = "https://hf-mirror.com"` 由脚本自动设置镜像环境变量。
- **本地目录里已有旧文件**：新生成的音频、字幕、日志会覆盖同名文件。
