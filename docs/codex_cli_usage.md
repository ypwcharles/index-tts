# Codex CLI 使用文档（关键用法）

> 适用版本：已在本机以 codex‑cli 0.40.0 验证。示例默认在项目根目录执行，或通过 `-C .` 指定工作目录为当前仓库。

## 快速上手

- 交互会话（无确认 + 关闭沙箱）

  ```bash
  codex --dangerously-bypass-approvals-and-sandbox -C . "请检查本仓库的 README 并提出改进建议"
  ```

- 单次执行（exec，非交互，完全自动）

  ```bash
  codex exec --dangerously-bypass-approvals-and-sandbox -C . "只需回复 OK 两个字。"
  ```

- 更安全的自动化（无需逐条确认）

  ```bash
  # 等价于：-a on-failure 且 -s workspace-write
  codex --full-auto -C . "分析本项目的目录结构并总结"
  ```

- 从标准输入读取 prompt（适合长指令/文件驱动）

  ```bash
  codex exec --dangerously-bypass-approvals-and-sandbox -C . - < prompt.txt
  ```

- 结构化与落盘输出

  ```bash
  codex exec --json --dangerously-bypass-approvals-and-sandbox -C . "..." \
    --output-last-message /tmp/last.txt
  ```

## 核心概念

- 模式：`codex` 进入交互会话；`codex exec` 进行非交互的单次任务执行。
- 确认策略：`-a never` 表示从不询问；`--full-auto` 是便捷配置（`-a on-failure` + `-s workspace-write`）。
- 沙箱策略：`-s read-only|workspace-write|danger-full-access`；完全放开用 `--dangerously-bypass-approvals-and-sandbox`（仅限可信/隔离环境）。
- 工作目录：用 `-C <DIR>` 设置工作根目录，建议始终限定为项目根（如 `-C .`）。
- 模型与提供方：可加 `-m <MODEL>`，默认读取本地配置（如 provider=openai, model=gpt-5）。

## 常见场景示例

- 定位包含 TODO 的文件并提出改进建议

  ```bash
  codex exec --full-auto -C . "在仓库中定位包含 TODO 的文件，并按优先级给出改进建议"
  ```

- 从文件驱动 prompt（长指令）

  ```bash
  codex exec --full-auto -C . - < tools/prompts/zh_localize_tts_prompt.md
  ```

- 附带图片进行分析

  ```bash
  codex exec --full-auto -i screenshot.png -C . "基于截图提出 UI 优化建议"
  ```

- 指定模型

  ```bash
  codex exec --full-auto -m gpt-4.1 -C . "请仅返回 OK"
  ```

- 仅保存最终回复（脚本/流水线友好）

  ```bash
  codex exec --full-auto -C . "..." --output-last-message /tmp/last.txt
  ```

## 认证与配置

- 版本与登录

  ```bash
  codex -V
  codex login status
  codex login --api-key <API_KEY>
  ```

- 覆盖配置键值（支持点号路径与 JSON 值）

  ```bash
  codex exec -C . \
    -c model="o3" \
    -c 'sandbox_permissions=["disk-full-read-access"]' \
    "请仅返回 OK"
  ```

- 使用配置档案（来自 `~/.codex/config.toml`）

  ```bash
  codex exec -p <PROFILE_NAME> -C . "..."
  ```

- 非 Git 目录（必要时）

  ```bash
  codex exec --skip-git-repo-check -C . "..."
  ```

## 脚本集成

- 最简单可靠的落盘方式

  ```bash
  codex exec --full-auto -C . "..." --output-last-message out.txt && cat out.txt
  ```

- 事件流处理（JSONL）

  ```bash
  codex exec --json --full-auto -C . "..." > events.jsonl
  ```

- 管道输入

  ```bash
  echo "请仅返回 OK" | codex exec --full-auto -C . - --output-last-message /tmp/last.txt
  ```

## 安全与最佳实践

- 优先使用 `--full-auto`（有沙箱、自动化）；仅在可信/隔离环境下使用 `--dangerously-bypass-approvals-and-sandbox`。
- 始终加 `-C .` 或在项目根目录执行，降低误操作风险。
- 在 CI 或生产脚本中使用 `--output-last-message` 稳定提取最终结果。
- 需要外部命令写权限时选择 `workspace-write`；只读审计场景用 `read-only`。

## 故障排查

- 帮助与版本

  ```bash
  codex help
  codex help exec
  codex -V
  ```

- 登录异常

  ```bash
  codex login status
  codex login --api-key <API_KEY>
  ```

- 标准输入未生效：确保使用独立的 `-` 占位符（例如 `... - < prompt.txt` 或通过管道并传入 `-`）。
- 非 Git 仓库：必要时使用 `--skip-git-repo-check`。
- 网络/配额：`--json` 事件流中会包含错误与配额信息，可据此定位。

## 已验证的关键命令（本机）

```bash
codex exec --dangerously-bypass-approvals-and-sandbox -C . "只需回复 OK 两个字。"

codex exec --json --dangerously-bypass-approvals-and-sandbox -C . "..." \
  --output-last-message /tmp/last.txt

codex exec --full-auto -C . "..." \
  --output-last-message /tmp/last_full_auto.txt
```

> 注：`--dangerously-bypass-approvals-and-sandbox` 会跳过所有确认并禁用沙箱，务必仅在可信环境中使用。

