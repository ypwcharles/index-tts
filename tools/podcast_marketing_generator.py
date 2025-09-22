#!/usr/bin/env python3
"""独立播客内容生成器

迁移自 personal/Translate 项目，调整为使用 index-tts 中的 OpenAI 兼容配置。
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from translate_openai_cli import (
    DEFAULT_ENV_PATH,
    build_endpoint,
    load_env_file,
    post_chat_completions,
)

# 独立播客生成器的提示词定义
PODCAST_CONTENT_GENERATION_ROLE = "You are a professional podcast content strategist and marketing expert specializing in creating compelling podcast content. You excel at extracting the most engaging moments from conversations and crafting irresistible marketing materials."

# 独立播客生成器的默认模型配置由命令行参数/env 控制


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    timeout: int = 300
    response_format: Optional[str] = "json_object"


class SubtitleProcessor:
    """字幕文件处理器"""
    
    @staticmethod
    def load_subtitle_json(file_path: str) -> Dict[str, Any]:
        """加载JSON字幕文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"❌ 加载字幕文件失败: {e}")
            raise
    
    @staticmethod
    def extract_text_with_timestamps(subtitle_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从字幕数据中提取文本和时间戳

        支持多种字幕格式：
        - segments: [{"start": 0.0, "end": 5.0, "text": "..."}]
        - transcription: {"segments": [...]}
        - results: [{"start": 0.0, "end": 5.0, "text": "..."}]
        - 直接数组格式
        """
        segments = []
        raw_segments = None

        # 尝试不同的数据结构
        if isinstance(subtitle_data, dict):
            # 尝试常见的键名
            possible_keys = [
                "segments",
                "results",
                "transcripts",
                "items",
                "subtitles",
            ]
            for key in possible_keys:
                if key in subtitle_data and isinstance(subtitle_data[key], list):
                    raw_segments = subtitle_data[key]
                    logging.info(f"🔍 找到数据在键 '{key}' 下，包含 {len(raw_segments)} 个元素")
                    break

            # 尝试嵌套结构
            if raw_segments is None:
                nested_keys = [
                    ("transcription", "segments"),
                    ("data", "segments"),
                    ("result", "segments"),
                    ("response", "segments")
                ]
                for parent_key, child_key in nested_keys:
                    if (parent_key in subtitle_data and
                        isinstance(subtitle_data[parent_key], dict) and
                        child_key in subtitle_data[parent_key] and
                        isinstance(subtitle_data[parent_key][child_key], list)):
                        raw_segments = subtitle_data[parent_key][child_key]
                        logging.info(f"🔍 找到数据在嵌套结构 '{parent_key}.{child_key}' 下，包含 {len(raw_segments)} 个元素")
                        break
        elif isinstance(subtitle_data, list):
            raw_segments = subtitle_data
            logging.info(f"🔍 数据为直接数组格式，包含 {len(raw_segments)} 个元素")

        if raw_segments is None:
            logging.error("❌ 无法在JSON中找到字幕数据")
            return segments

        # 处理每个段落
        for i, segment in enumerate(raw_segments):
            if not isinstance(segment, dict):
                logging.warning(f"⚠️ 跳过非字典格式的段落 {i}: {type(segment)}")
                continue

            # 尝试不同的字段名
            text_fields = ["text", "content", "transcript", "sentence"]
            start_fields = ["start", "start_time", "time_begin", "begin", "from"]
            end_fields = ["end", "end_time", "time_end", "finish", "to"]

            text = None
            start_time = None
            end_time = None

            # 查找文本字段
            for field in text_fields:
                if field in segment and segment[field]:
                    text = str(segment[field]).strip()
                    break

            # 查找开始时间字段
            for field in start_fields:
                if field in segment and segment[field] is not None:
                    try:
                        time_value = float(segment[field])
                        # 如果时间值很大（>1000），可能是毫秒，需要转换为秒
                        if time_value > 1000:
                            start_time = time_value / 1000.0
                        else:
                            start_time = time_value
                        break
                    except (ValueError, TypeError):
                        continue

            # 查找结束时间字段
            for field in end_fields:
                if field in segment and segment[field] is not None:
                    try:
                        time_value = float(segment[field])
                        # 如果时间值很大（>1000），可能是毫秒，需要转换为秒
                        if time_value > 1000:
                            end_time = time_value / 1000.0
                        else:
                            end_time = time_value
                        break
                    except (ValueError, TypeError):
                        continue

            # 如果找到了文本和开始时间，添加到结果中
            if text and start_time is not None:
                segments.append({
                    "text": text,
                    "start": start_time,
                    "end": end_time if end_time is not None else start_time,
                    "timestamp": SubtitleProcessor.format_timestamp(start_time)
                })
            else:
                logging.debug(f"⚠️ 跳过段落 {i}，缺少必要字段 - text: {bool(text)}, start: {start_time is not None}")

        return segments
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """将秒数转换为时间戳格式 (MM:SS)"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def combine_segments_to_text(segments: List[Dict[str, Any]]) -> str:
        """将分段文本合并为完整文本"""
        return " ".join([seg["text"] for seg in segments if seg["text"]])


class EnhancedPodcastContentGenerator:
    """增强版播客内容生成器，支持时间戳"""

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.endpoint = build_endpoint(llm_config.base_url)

        # 初始化 token 统计
        self.total_tokens = {
            "prompt": 0,
            "completion": 0,
            "total": 0,
        }

    def _update_token_counts(self, token_counts: Dict[str, int]):
        """更新累计token计数"""
        for key in self.total_tokens:
            if key == "completion":
                # 映射completion_tokens到completion
                self.total_tokens[key] += token_counts.get("completion_tokens", 0)
            else:
                self.total_tokens[key] += token_counts.get(key, 0)

    def _display_usage_summary(self):
        """显示token使用情况和成本总结"""
        logging.info("=" * 60)
        logging.info("📊 播客内容生成统计信息")
        logging.info("=" * 60)
        logging.info(f"🔢 Token使用情况:")
        logging.info(f"   - 输入Token: {self.total_tokens['prompt']:,}")
        logging.info(f"   - 输出Token: {self.total_tokens['completion']:,}")
        logging.info(f"   - 总Token: {self.total_tokens['total']:,}")
        logging.info("💰 估算成本: 暂未计算（可结合实际定价手动估算）")
        logging.info("=" * 60)
        
    def generate_content_from_subtitle(self, subtitle_file: Path, output_dir: Path) -> Optional[Path]:
        """从字幕文件生成播客内容"""

        # 1. 加载和处理字幕文件
        logging.info(f"📁 加载字幕文件: {subtitle_file}")
        subtitle_data = SubtitleProcessor.load_subtitle_json(str(subtitle_file))

        # 调试：显示JSON文件的结构
        logging.info(f"🔍 JSON文件结构调试:")
        logging.info(f"   - 顶层键: {list(subtitle_data.keys()) if isinstance(subtitle_data, dict) else '不是字典类型'}")
        if isinstance(subtitle_data, dict):
            for key, value in subtitle_data.items():
                if isinstance(value, list):
                    logging.info(f"   - {key}: 列表，长度 {len(value)}")
                    if len(value) > 0:
                        logging.info(f"     首个元素类型: {type(value[0])}")
                        if isinstance(value[0], dict):
                            logging.info(f"     首个元素键: {list(value[0].keys())}")
                elif isinstance(value, dict):
                    logging.info(f"   - {key}: 字典，键: {list(value.keys())}")
                else:
                    logging.info(f"   - {key}: {type(value)}")

        segments = SubtitleProcessor.extract_text_with_timestamps(subtitle_data)

        logging.info(f"✅ 成功提取 {len(segments)} 个文本段落")

        # 如果没有提取到任何段落，停止处理
        if len(segments) == 0:
            logging.error("❌ 未能从字幕文件中提取到任何文本段落，请检查文件格式")
            logging.error("💡 支持的格式示例:")
            logging.error("   1. {\"segments\": [{\"start\": 0.0, \"end\": 5.0, \"text\": \"...\"}]}")
            logging.error("   2. {\"transcription\": {\"segments\": [...]}}")
            logging.error("   3. [{\"start\": 0.0, \"end\": 5.0, \"text\": \"...\"}]")
            return None

        # 计算音频总时长
        total_duration = 0
        if segments:
            total_duration = max(seg.get('end', seg.get('start', 0)) for seg in segments)

        # 格式化总时长
        total_duration_formatted = self._format_duration(total_duration)

        logging.info(f"🕒 音频总时长: {total_duration_formatted} ({total_duration:.1f}秒)")

        # 2. 生成播客内容
        logging.info("🎯 开始生成播客营销内容...")
        enhanced_prompt = self._create_enhanced_prompt(segments, total_duration, total_duration_formatted)
        
        # 调用LLM生成内容
        response = self._call_llm_for_content_generation(enhanced_prompt)
        
        if not response:
            logging.error("❌ 播客内容生成失败")
            return None
        
        # 3. 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"podcast_content_{timestamp}.md"

        # 创建综合内容文档（包含纯文本Show Notes）
        comprehensive_content = self._create_comprehensive_content_with_plain_show_notes(response)
        
        with output_file.open('w', encoding='utf-8') as f:
            f.write(comprehensive_content)

        logging.info(f"✅ 播客内容已保存到: {output_file}")

        # 验证生成的时间轴完整性
        if response and 'show_notes' in response and 'timeline' in response['show_notes']:
            self._validate_timeline_completeness(response['show_notes']['timeline'], total_duration, total_duration_formatted)

        # 显示token使用情况和成本统计
        self._display_usage_summary()

        return output_file
    
    def _create_enhanced_prompt(self, segments: List[Dict[str, Any]], total_duration: float, total_duration_formatted: str) -> str:
        """创建增强的提示，包含时间戳信息"""

        # 创建专业级Show Notes提示词
        professional_shownotes_prompt = self._create_professional_shownotes_prompt(segments, total_duration, total_duration_formatted)

        return professional_shownotes_prompt

    def _format_duration(self, seconds: float) -> str:
        """将秒数转换为时长格式 (HH:MM:SS 或 MM:SS)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def _validate_timeline_completeness(self, timeline: List[Dict[str, Any]], total_duration: float, total_duration_formatted: str):
        """验证生成的时间轴是否覆盖了完整的音频时长"""
        if not timeline:
            logging.warning("⚠️ 时间轴为空")
            return

        # 解析时间轴中的最后一个时间点
        last_timeline_entry = timeline[-1]
        last_time_str = last_timeline_entry.get('time', '00:00')

        try:
            # 解析时间字符串 (MM:SS 或 HH:MM:SS)
            time_parts = last_time_str.split(':')
            if len(time_parts) == 2:  # MM:SS
                last_time_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
            elif len(time_parts) == 3:  # HH:MM:SS
                last_time_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
            else:
                logging.warning(f"⚠️ 无法解析时间格式: {last_time_str}")
                return

            # 计算覆盖率
            coverage_percentage = (last_time_seconds / total_duration) * 100 if total_duration > 0 else 0

            logging.info(f"📊 时间轴覆盖情况:")
            logging.info(f"   - 字幕文件总时长: {total_duration_formatted}")
            logging.info(f"   - 时间轴最后时间点: {last_time_str}")
            logging.info(f"   - 覆盖率: {coverage_percentage:.1f}%")

            # 更严格的覆盖率检查
            if coverage_percentage < 70:
                logging.error(f"❌ 时间轴覆盖率过低 ({coverage_percentage:.1f}%)，严重不足")
                logging.error(f"💡 时间轴未能覆盖字幕文件的大部分内容，建议重新生成")
            elif coverage_percentage < 85:
                logging.warning(f"⚠️ 时间轴覆盖率偏低 ({coverage_percentage:.1f}%)，可能遗漏重要内容")
                logging.warning(f"💡 建议检查生成的时间轴是否包含字幕文件结尾部分的内容")
            elif coverage_percentage >= 90:
                logging.info(f"✅ 时间轴覆盖率优秀 ({coverage_percentage:.1f}%)")
            else:
                logging.info(f"✅ 时间轴覆盖率良好 ({coverage_percentage:.1f}%)")

        except (ValueError, IndexError) as e:
            logging.warning(f"⚠️ 时间轴验证失败: {e}")
            logging.warning(f"   最后时间点: {last_time_str}")

    def _create_professional_shownotes_prompt(self, segments: List[Dict[str, Any]], total_duration: float, total_duration_formatted: str) -> str:
        """创建专业级Show Notes生成提示词，基于Acquired播客风格"""

        # 使用完整的segments数据，不进行取样
        total_segments = len(segments)

        logging.info(f"📊 将向AI提供完整的字幕文件：{total_segments} 个字幕段落用于分析")

        # 将完整的segments转换为JSON格式字符串，用于提示词
        segments_json = json.dumps(segments, ensure_ascii=False, indent=2)

        return f"""# === 角色与核心目标定义 ===
你是一位顶尖的商业与科技播客制作人兼撰稿人，为一档风格类似于《Acquired》的中文播客《他山之声》工作。《他山之声》是一档专门编译全球优秀外语播客内容的节目，致力于为中文听众带来全球最前沿的思考和对话。你的核心任务是分析所提供的JSON格式播客字幕文件，并生成一套全面、富有洞察力且由叙事驱动的播客营销内容，包括钩子、内容概要、标题候选和专业级Show Notes（含详细的带时间戳时间轴）。

# === 字幕文件信息 ===
**字幕文件时间范围**: 00:00 - {total_duration_formatted} (总计{total_duration:.1f}秒)
**提供的字幕段落数**: {total_segments} (完整字幕文件)
**重要**: 你必须确保生成的时间轴覆盖从00:00到接近{total_duration_formatted}的完整时间范围

# === 输入数据结构定义 ===
输入是一个JSON对象，包含字幕片段列表。每个片段包含："text"（字符串，文本内容）、"start"（浮点数，开始时间秒）、"end"（浮点数，结束时间秒）和"timestamp"（字符串，格式化的时间戳）。

# === 钩子生成的关键要求 ===
**重要：钩子必须完全引用原始JSON字幕文件中的原话**
- 钩子的"text"字段必须是从JSON字幕片段中完全复制的原始文本，不允许任何改写、重组或修改
- 可以选择单个完整的字幕片段，或者选择连续的多个字幕片段进行组合
- 但绝对不能对原始文本进行任何编辑、改写或重新表述
- 时间戳必须准确对应所选择的字幕片段的实际时间戳
- 选择标准：原话必须对观众极具吸引力，能够激发好奇心或提供震撼性洞察

# === Show Notes语气和定位要求 ===
**重要：《他山之声》节目定位和语气要求**
- 《他山之声》是一档编译节目，所有内容都来自其他播客的编译和转述
- 节目简介必须体现编译性质，使用客观转述的语气，避免第一人称表述
- 正确示例："本期《他山之声》为你带来《Big Technology Podcast》的一档节目，与Spyglass记者Mg Glerum一起，深度拆解科技巨头在AI时代的战略、权谋与失控瞬间"
- 错误示例："本期《他山之声》与Spyglass记者Mg Glerum一起，深度拆解科技巨头在AI时代的战略、权谋与失控瞬间"
- 嘉宾简介也要保持客观转述的语气，说明这是来自原播客的嘉宾
- Show Notes内容要去除Markdown格式标记，确保在小宇宙等平台能正常显示

# === 链式思考（Chain-of-Thought）工作流指令 ===
为了完成这个任务，你必须严格遵循以下分步流程。在完成所有内部思考步骤之前，不要输出最终结果。让我们一步一步来思考：

**第一步：整合字幕 (Consolidate Transcript)**
通读整个JSON片段列表。将连续的相关片段合并成更大的段落，生成一份干净、可读的对话脚本。

**第二步：识别主题边界 (Identify Thematic Boundaries)**
仔细阅读整合后的对话脚本，识别主要的叙事或主题转换点。标记以下迹象的新章节边界：
- 主持人明确转换话题
- 重要新概念、人物、公司或事件被引入
- 叙事时间线出现显著跳跃
- 快速问答后出现长解释性独白

**第三步：起草章节大纲 (Draft Segment Outline)**
根据识别的边界，创建章节大纲草稿。对每个章节列出起始时间、结束时间，并用不超过15个词概括核心主题。
**重要：确保时间轴覆盖完整字幕文件时长**
- 时间轴必须从00:00开始，一直覆盖到接近字幕文件结束时间{total_duration_formatted}
- 最后一个时间点应该在{total_duration_formatted}前的最后几分钟内
- 你已经获得了完整的字幕文件内容，请确保时间轴覆盖整个对话的完整发展过程

**第四步：识别最具吸引力的原始引用 (Identify Most Compelling Original Quotes)**
仔细审查JSON字幕片段，寻找最具吸引力的原始话语作为钩子候选。这些话语必须：
- 完全来自原始字幕片段，不做任何修改
- 具有强烈的吸引力、争议性或洞察性
- 能够激发听众的好奇心
- 可以是单个片段或连续的多个片段

**第五步：验证时间轴完整性 (Verify Timeline Completeness)**
检查生成的时间轴是否满足以下要求：
- 从00:00开始
- 最后一个时间点接近字幕文件结束时间{total_duration_formatted}
- 时间点分布合理，没有大的时间空隙
- 确保时间轴覆盖率达到90%以上（即最后时间点应在{total_duration_formatted}的90%以后）
- 基于完整的字幕文件内容，确保时间轴反映对话的完整发展脉络

**第六步：生成营销内容 (Generate Marketing Content)**
基于分析生成钩子、概要、标题和Show Notes，风格要引人入胜且富有洞察力。钩子必须严格遵循第四步的要求。

# === 少样本学习范例 ===
**Show Notes时间轴风格范例：**

### 范例 1
(00:12:45) 坎普拉德家族的德国渊源与一次宿命般的购买
宜家的故事并非始于瑞典，而是德国。主持人详细讲述了英格瓦·坎普拉德的祖父母如何因为家庭反对他们的婚姻，而从德国移民至瑞典一个艰苦的农业区。他们通过一本杂志广告，在未曾亲眼见过的情况下买下了一个农场。这次迁移在家族内部种下了一颗深刻的文化种子：一种摆脱困境、创造财富的强烈驱动力。

### 范例 2
(00:03:10) 理解Meta前所未有的规模
Ben和David在开篇便通过数据揭示了Meta惊人的用户规模——40亿月活跃用户，接近全球人口的一半。他们将其与历史上最庞大的帝国和政府进行比较，得出的结论是：没有任何一个机构曾连接过如此高比例的全球人口。

# === 输入数据 ===
**字幕数据JSON：**
```json
{segments_json}
```

# === 最终输出格式规范 ===
你的最终输出必须是一个JSON对象，格式如下：

```json
{{
  "hooks": {{
    "option_1": {{
      "sentences": [
        {{
          "text": "完全来自原始JSON字幕片段的原话，不允许任何修改",
          "timestamp": "MM:SS",
          "reason": "选择这段原话的理由（为什么这段原话具有吸引力）"
        }}
      ],
      "explanation": "为什么选择这个组合的说明，强调这些是完全未经修改的原始引用"
    }},
    "option_2": {{ ... }},
    "option_3": {{ ... }}
  }},
  "content_overview": "欢迎收听《他山之声》，一档致力于打破语言壁垒，为你带来全球最前沿思考的播客节目。我是你的主播，查哥查尔斯。[继续150-200字的内容概要]",
  "title_candidates": [
    {{
      "title": "标题1",
      "reason": "选择理由"
    }}
  ],
  "show_notes": {{
    "episode_intro": "节目简介（100-150字），必须体现编译性质，使用'本期《他山之声》为你带来《原播客名称》的一档节目'的格式开头，保持客观转述语气",
    "guest_intro": "嘉宾介绍（100-150字），客观介绍原播客的嘉宾，避免第一人称表述",
    "key_points": ["要点1", "要点2", "要点3", "要点4", "要点5"],
    "timeline": [
      {{
        "time": "00:00",
        "topic": "引人入胜的章节标题",
        "description": "富有洞察力的摘要，说明听众能获得什么具体价值、洞见或行动指导（50-100字）",
        "section": "第一部分：主题名称（可选）"
      }},
      {{
        "time": "XX:XX",
        "topic": "中间章节标题",
        "description": "继续提供有价值的内容描述",
        "section": "第二部分：主题名称（可选）"
      }},
      {{
        "time": "接近{total_duration_formatted}的时间点",
        "topic": "结尾章节标题",
        "description": "确保时间轴覆盖到音频结尾附近",
        "section": "最后部分：主题名称（可选）"
      }}
    ],
    "additional_notes": "关于原节目：本期内容编译自知名外语播客节目...\\n\\n技术说明：本节目使用AI技术对原有节目人声进行克隆和语音合成后制作而成，可能在某些语句的发音并不能做到完全自然。\\n\\n听友交流：如果有想要听的其他外语播客或者节目，欢迎联系微信：chagexq，进入听友群。"
  }}
}}
```

请严格按照上述格式输出，确保时间轴部分采用类似Acquired播客的深度分析风格，每个时间点都要有引人入胜的标题和富有洞察力的描述。

**最重要的提醒：时间轴完整性要求**
- 时间轴必须从00:00开始，覆盖到接近字幕文件结束时间{total_duration_formatted}
- 最后一个时间点应该在{total_duration_formatted}前的最后几分钟内
- 时间点分布要合理，避免大的时间空隙
- 你已经获得了完整的字幕文件内容，请确保时间轴反映整个对话的完整发展过程
- 覆盖率必须达到90%以上，即最后时间点应在总时长的90%以后

**钩子生成要求**
- 钩子的"text"字段必须是从提供的JSON字幕片段中完全复制的原始文本
- 绝对不允许对原始文本进行任何改写、重组、编辑或修改
- 可以选择单个完整片段或连续的多个片段，但文本内容必须保持完全一致
- 时间戳必须准确对应所选择片段的实际时间戳
- 选择的原话必须对观众极具吸引力，能够激发好奇心或提供震撼性洞察"""

    def _call_llm_for_content_generation(self, prompt: str) -> Optional[Dict[str, Any]]:
        """调用LLM生成播客内容"""
        payload: Dict[str, Any] = {
            "model": self.llm_config.model,
            "messages": [
                {"role": "system", "content": PODCAST_CONTENT_GENERATION_ROLE},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.llm_config.temperature,
            "top_p": self.llm_config.top_p,
        }
        if self.llm_config.max_tokens is not None:
            payload["max_tokens"] = self.llm_config.max_tokens
        if self.llm_config.response_format:
            payload["response_format"] = {"type": self.llm_config.response_format}

        try:
            response = post_chat_completions(
                self.endpoint,
                self.llm_config.api_key,
                payload,
                timeout=self.llm_config.timeout,
            )
        except SystemExit as exc:  # translate_openai_cli raises SystemExit on HTTP errors
            logging.error(f"❌ LLM调用失败: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            logging.error(f"❌ LLM调用失败: {exc}")
            return None

        logging.debug(f"LLM响应结构: {response}")

        usage = response.get("usage")
        if usage:
            token_counts = {
                "prompt": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }
            self._update_token_counts(token_counts)
            logging.info(
                "  - Token使用: %s 输入 + %s 输出 = %s 总计",
                token_counts["prompt"],
                token_counts["completion_tokens"],
                token_counts["total"],
            )
        else:
            logging.warning("  - 无token使用信息可用")

        choices = response.get("choices") or []
        if not choices:
            logging.error("❌ 未收到有效响应 (choices 为空)")
            return None

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            logging.error("❌ 未收到有效响应内容")
            return None

        logging.debug(f"原始响应内容: {content[:500]}...")

        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end != -1:
                content = content[json_start:json_end].strip()
                logging.debug(f"提取的JSON内容: {content[:200]}...")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logging.error(f"❌ JSON解析失败: {exc}")
            logging.error(f"尝试解析的内容: {content[:1000]}")
            return None

    def _create_comprehensive_content_with_plain_show_notes(self, podcast_content: Dict[str, Any]) -> str:
        """创建包含纯文本Show Notes的综合内容文档"""
        if not podcast_content:
            return "# 播客内容生成失败\n\n无法生成播客内容，请检查输入数据。"

        content = []
        content.append("# 《他山之声》播客内容包")
        content.append("")
        content.append("本文档包含本期播客的完整营销内容，包括钩子、内容概要、标题候选和Show Notes。")
        content.append("")
        content.append("---")
        content.append("")

        # 1. 播客钩子
        content.append("## 1️⃣ 播客钩子 (Hooks)")
        content.append("")
        content.append("**使用说明：** 以下是3组钩子组合，可根据目标受众和平台特点选择使用。")
        content.append("")

        hooks = podcast_content.get("hooks", {})
        for i, (_, hook_data) in enumerate(hooks.items(), 1):
            content.append(f"### 组合 {i}")
            content.append("")

            sentences = hook_data.get("sentences", [])
            for sentence_data in sentences:
                if isinstance(sentence_data, dict):
                    text = sentence_data.get("text", "")
                    timestamp = sentence_data.get("timestamp", "")
                    reason = sentence_data.get("reason", "")

                    if timestamp:
                        content.append(f"- **[{timestamp}]** {text}")
                    else:
                        content.append(f"- {text}")

                    if reason:
                        content.append(f"  *选择理由：{reason}*")
                else:
                    content.append(f"- {sentence_data}")
                content.append("")

            explanation = hook_data.get("explanation", "")
            if explanation:
                content.append(f"**为什么这个组合有效：** {explanation}")
                content.append("")

        content.append("---")
        content.append("")

        # 2. 内容概要
        content.append("## 2️⃣ 内容概要 (Content Overview)")
        content.append("")
        content.append("**使用说明：** 此内容概要放在钩子之后、正式播客内容之前。")
        content.append("")

        overview = podcast_content.get("content_overview", "")
        if overview:
            content.append(overview)
        else:
            content.append("内容概要生成失败。")

        content.append("")
        content.append("---")
        content.append("")

        # 3. 标题候选
        content.append("## 3️⃣ 标题候选 (Title Candidates)")
        content.append("")
        content.append("**使用说明：** 以下是5个标题候选，可根据平台特点和目标受众选择使用。")
        content.append("")

        titles = podcast_content.get("title_candidates", [])
        for i, title_data in enumerate(titles, 1):
            if isinstance(title_data, dict):
                title = title_data.get("title", "")
                reason = title_data.get("reason", "")

                content.append(f"### 标题 {i}")
                content.append("")
                content.append(f"**{title}**")
                content.append("")
                if reason:
                    content.append(f"**选择理由：** {reason}")
                content.append("")
            else:
                content.append(f"### 标题 {i}")
                content.append("")
                content.append(f"**{title_data}**")
                content.append("")

        content.append("---")
        content.append("")

        # 4. Show Notes (纯文本版本)
        content.append("## 4️⃣ Show Notes（纯文本版本，适用于小宇宙等平台）")
        content.append("")
        content.append("**使用说明：** 以下Show Notes已去除Markdown格式，可直接复制粘贴到小宇宙等播客平台。")
        content.append("")
        content.append("```")

        show_notes = podcast_content.get("show_notes", {})

        # 节目简介
        episode_intro = show_notes.get("episode_intro", "")
        if episode_intro:
            content.append("🎙️ 本期节目简介")
            content.append("")
            content.append(episode_intro)
            content.append("")

        # 嘉宾介绍
        guest_intro = show_notes.get("guest_intro", "")
        if guest_intro:
            content.append("🧑‍💻 本期嘉宾简介")
            content.append("")
            content.append(guest_intro)
            content.append("")

        # 核心要点
        key_points = show_notes.get("key_points", [])
        if key_points:
            content.append("🔑 本期核心要点")
            content.append("")
            for point in key_points:
                content.append(f"• {point}")
            content.append("")

        # 时间轴
        timeline = show_notes.get("timeline", [])
        if timeline:
            content.append("⏰ 时间轴")
            content.append("")
            for item in timeline:
                if isinstance(item, dict):
                    time = item.get("time", "")
                    topic = item.get("topic", "")
                    description = item.get("description", "")

                    content.append(f"{time} {topic}")
                    if description:
                        content.append(description)
                    content.append("")

        # 补充说明
        additional_notes = show_notes.get("additional_notes", "")
        if additional_notes:
            content.append("📝 补充说明")
            content.append("")
            content.append(additional_notes)
            content.append("")

        content.append("```")
        content.append("")
        content.append("---")
        content.append("")
        content.append("*本内容由AI自动生成，请根据实际需要进行调整和优化。*")

        return "\n".join(content)


def main() -> int:
    """主函数"""
    parser = argparse.ArgumentParser(description="独立播客内容生成器")
    parser.add_argument("subtitle_file", type=Path, help="JSON字幕文件路径")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出目录 (默认: 与字幕文件同路径下的 podcast_marketing_outputs)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="OpenAI 兼容服务配置 env 文件 (默认: tools/openai_translator.env)",
    )
    parser.add_argument("--base-url", help="覆盖 env 中的 OPENAI_BASE_URL")
    parser.add_argument("--api-key", help="覆盖 env 中的 OPENAI_API_KEY")
    parser.add_argument("--model", help="覆盖 env 中的 OPENAI_MODEL")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=int, default=300, help="HTTP 超时（秒）")
    parser.add_argument(
        "--response-format",
        default="json_object",
        help="OpenAI response_format.type，填 none 关闭",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志输出")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    subtitle_path = args.subtitle_file.expanduser().resolve()
    if not subtitle_path.is_file():
        logging.error(f"❌ 字幕文件不存在: {subtitle_path}")
        return 1

    if args.output is not None:
        output_dir = args.output.expanduser().resolve()
    else:
        output_dir = (subtitle_path.parent / "podcast_marketing_outputs").resolve()
    env_path = args.env.expanduser().resolve()

    env_vars = load_env_file(env_path)

    base_url = args.base_url or env_vars.get("OPENAI_BASE_URL") or "https://api.openai.com"
    api_key = args.api_key or env_vars.get("OPENAI_API_KEY") or ""
    model = args.model or env_vars.get("OPENAI_MODEL") or "gpt-4o-mini"

    response_format = args.response_format
    if response_format:
        if str(response_format).lower() in {"none", "null", "off"}:
            response_format_value: Optional[str] = None
        else:
            response_format_value = str(response_format)
    else:
        response_format_value = None

    if not api_key:
        logging.warning("OPENAI_API_KEY 未配置（env 或 --api-key）。若目标服务需要鉴权，请设置。")

    llm_config = LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        response_format=response_format_value,
    )

    try:
        generator = EnhancedPodcastContentGenerator(llm_config)
        output_file = generator.generate_content_from_subtitle(subtitle_path, output_dir)

        if output_file:
            print("\n✅ 播客内容生成完成！")
            print(f"📁 输出文件: {output_file}")
            return 0

        print("\n❌ 播客内容生成失败，请查看日志获取详细信息")
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.error(f"❌ 程序执行失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
