#!/usr/bin/env python3
"""
Extract precise audio subclips from a WhisperX word-level JSON by matching
Chinese/English phrases, then stitch them into a single MP3 via ffmpeg.

Usage:
  python tools/cut_podcast_clips.py \
    --json /path/to/0918ama.json \
    --audio /path/to/0918ama.MP3 \
    --out /path/to/output.mp3 \
    [--dry-run]

Notes:
  - Matches are done against the per-segment `words` list by concatenating
    each `word` field; this is robust for Chinese (char-level timestamps).
  - Phrases are tried in order; the first match in the recommended time window
    wins. If no match is found in the window, the whole window is used.
  - Produces a single ffmpeg invocation using atrim + concat for clean cuts.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PhraseSpec:
    # Approximate time window to search within (seconds)
    t_min: float
    t_max: float
    # Candidate phrases to look for (first match wins)
    phrases: List[str]
    # Optional label for reporting
    label: str = ""


def load_whisperx(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_segment_word_strings(segments: List[Dict]) -> List[Dict]:
    out = []
    for s in segments:
        words = s.get("words") or []
        # Concatenate words (characters) and prepare lowercased version for search
        chars = [w.get("word", "") for w in words]
        s_concat = "".join(chars)
        out.append({
            "start": s["start"],
            "end": s["end"],
            "text": s.get("text", ""),
            "words": words,
            "concat": s_concat,
            "concat_lower": s_concat.lower(),
        })
    return out


def find_phrase_in_segment(seg: Dict, phrase: str) -> Optional[Tuple[float, float]]:
    """Return (start, end) seconds for the first exact match of phrase in this segment.

    We search inside the concatenated word characters. Matching is exact on that
    string (after lowercasing both), which is reliable for Chinese and roman tokens.
    """
    concat = seg["concat_lower"]
    target = phrase.lower()
    idx = concat.find(target)
    if idx < 0:
        return None
    words = seg["words"]
    # Map character indices to word timestamps
    # Note: Some punctuation may not be present in `words`; we matched in concat built from words.
    start_w = words[idx]
    end_w = words[idx + len(target) - 1]
    return float(start_w["start"]), float(end_w["end"])


def find_phrase_in_window(
    segs: List[Dict], spec: PhraseSpec
) -> Optional[Tuple[float, float, str]]:
    """Search for spec.phrases within segments overlapping [t_min, t_max].
    Returns (start, end, matched_phrase) or None.

    Preference order:
      1) Higher-priority phrases earlier in the list
      2) Earliest occurrence in time for that phrase
    """
    # Iterate phrases first to honor priority, then find the earliest segment match
    for phrase in spec.phrases:
        for seg in segs:
            if seg["end"] < spec.t_min or seg["start"] > spec.t_max:
                continue
            r = find_phrase_in_segment(seg, phrase)
            if r is not None:
                return r[0], r[1], phrase
    return None


def ffmpeg_concat_filter(clips: Sequence[Tuple[float, float]]) -> str:
    """Build a filter_complex string that trims [0:a] for each (start,end) and concats them."""
    parts = []
    labels = []
    for i, (start, end) in enumerate(clips):
        parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[a{i}]")
    n = len(clips)
    parts.append(f"{''.join(labels)}concat=n={n}:v=0:a=1[outa]")
    return ";".join(parts)


def run_ffmpeg(audio: Path, out: Path, clips: Sequence[Tuple[float, float]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    filt = ffmpeg_concat_filter(clips)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio),
        "-filter_complex", filt,
        "-map", "[outa]",
        "-c:a", "mp3",
        "-b:a", "192k",
        str(out),
    ]
    print("\nFFmpeg command:\n", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = load_whisperx(args.json)
    segs = build_segment_word_strings(data.get("segments", []))

    # Helper to convert mm:ss approximate to seconds
    def mmss(s: str) -> float:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)

    # Define the 8 clip points with phrases and windows
    # Windows are generous around the user's hints to ensure matches.
    specs: List[List[PhraseSpec]] = [
        # Clip 1: three snippets
        [
            PhraseSpec(t_min=mmss("07:45"), t_max=mmss("09:20"), phrases=["黄金在不断的涨", "黄金在不断的涨的背景下", "黄金在不断的涨的"], label="黄金在不断的涨"),
            PhraseSpec(t_min=mmss("07:55"), t_max=mmss("09:30"), phrases=["各国政府现在买美国国债的意愿", "买美国国债的意愿"], label="各国政府买美国国债的意愿"),
            PhraseSpec(t_min=mmss("08:30"), t_max=mmss("09:30"), phrases=["你会怎么解决", "你会怎么解决总之"], label="你会怎么解决"),
        ],
        # Clip 2: two snippets
        [
            PhraseSpec(t_min=mmss("09:05"), t_max=mmss("09:30"), phrases=["答案很简单", "其实答案很简单"], label="答案很简单"),
            PhraseSpec(t_min=mmss("09:05"), t_max=mmss("09:40"), phrases=["政府不买散户可以买", "政府不买就让散户买"], label="政府不买散户可以买"),
        ],
        # Clip 3: two snippets
        [
            PhraseSpec(t_min=mmss("09:10"), t_max=mmss("10:10"), phrases=["USDT的成功其实就是代表了全世界对美元有多么的渴望", "TetherUSDT的成功其实就是代表了全世界对美元有多么的渴望", "USDT的成功", "对美元有多么的渴望"], label="USDT的成功/渴望"),
            PhraseSpec(t_min=mmss("09:20"), t_max=mmss("10:20"), phrases=["给了财长和贝森特和川普很大的启发", "给了川普很大的启发", "很大的启发"], label="给了川普很大的启发"),
        ],
        # Clip 4: two snippets
        [
            PhraseSpec(t_min=mmss("11:20"), t_max=mmss("12:30"), phrases=["怎么让全世界的散户去大量的买入稳定币", "让全世界的散户去大量的买入稳定币"], label="散户大量买入稳定币"),
            PhraseSpec(t_min=mmss("11:30"), t_max=mmss("12:40"), phrases=["都会成为superapp", "都会成为super app", "都会成为superAPP"], label="成为super app"),
        ],
        # Clip 5: two snippets (both may be in one segment)
        [
            PhraseSpec(t_min=mmss("12:40"), t_max=mmss("13:20"), phrases=["稳定币支付框架", "谷歌发行了一个稳定币支付框架"], label="稳定币支付框架"),
            PhraseSpec(t_min=mmss("12:40"), t_max=mmss("13:30"), phrases=["链上的swift", "链上的swift吧"], label="链上swift"),
        ],
        # Clip 6: coercion/resistance core idea
        [
            PhraseSpec(t_min=mmss("14:50"), t_max=mmss("15:40"), phrases=["很多国家其实会抵制", "处处防着美国金融巨头进入他当地的市场"], label="抵制/防范"),
            PhraseSpec(t_min=mmss("14:50"), t_max=mmss("15:40"), phrases=["霸凌的这种姿态去推动", "霸凌的姿态"], label="霸凌姿态推动"),
        ],
        # Clip 7: one main snippet + optional question snippet
        [
            PhraseSpec(t_min=mmss("17:20"), t_max=mmss("18:20"), phrases=["钱一旦进入了稳定币", "进入了稳定币"], label="钱进入稳定币"),
            PhraseSpec(t_min=mmss("17:20"), t_max=mmss("18:30"), phrases=["离开了传统金融的这个体系", "离开了传统金融的体系"], label="离开传统金融体系"),
            # Rhetorical question about最大的两家输家（来自~11:32）。
            PhraseSpec(t_min=mmss("11:20"), t_max=mmss("11:55"), phrases=["最大的两家输家", "两大输家"], label="两大输家提问"),
        ],
        # Clip 8: composite of three snippets
        [
            PhraseSpec(t_min=mmss("18:10"), t_max=mmss("18:50"), phrases=["为什么说这是一个撕裂的漫牛", "撕裂的漫牛"], label="为什么撕裂的慢牛"),
            PhraseSpec(t_min=mmss("22:20"), t_max=mmss("23:50"), phrases=["一边是追求正式收入高回报", "另外一面", "土狗在炒"], label="一边价值一边土狗"),
            PhraseSpec(t_min=mmss("00:25"), t_max=mmss("00:55"), phrases=["看懂聪明前如何下注抓住下一个十倍机遇", "看懂聪明前如何下注", "抓住下一个十倍机遇"], label="聪明钱与十倍机遇"),
        ],
    ]

    resolved_clips: List[Tuple[float, float, str, str]] = []  # (start, end, label, phrase)
    for group_idx, group in enumerate(specs, start=1):
        for spec in group:
            match = find_phrase_in_window(segs, spec)
            if match is not None:
                start, end, phrase = match
                resolved_clips.append((start, end, f"clip{group_idx}:{spec.label}", phrase))
            else:
                # Fallback: use the window directly
                resolved_clips.append((spec.t_min, spec.t_max, f"clip{group_idx}:{spec.label}", "<window>"))

    # Merge very short adjacent snippets from same segment/phrase window? Keep simple: leave as-is

    # Sort by time to maintain requested order (we already append in order)
    # But we want the logical order specified, so keep insertion order.

    # Report
    print("Resolved subclips (start -> end, label | phrase):")
    total_dur = 0.0
    for s, e, label, phrase in resolved_clips:
        dur = e - s
        total_dur += dur
        print(f"  {s:8.3f} -> {e:8.3f}  ({dur:6.3f}s)  {label} | {phrase}")
    print(f"Total duration: {total_dur:.3f}s  ({total_dur/60:.2f} min)")

    clips_for_ffmpeg = [(s, e) for s, e, _, _ in resolved_clips]

    if args.dry_run:
        print("\nDry run: not invoking ffmpeg.")
        return

    run_ffmpeg(args.audio, args.out, clips_for_ffmpeg)


if __name__ == "__main__":
    main()
