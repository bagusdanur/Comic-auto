#!/usr/bin/env python3
"""Typeset Engine V4: Shape-aware comic typesetting for manga/webtoon bubbles.

Key capabilities:
1. Diamond / Ellipse text profile matching speech balloon geometry.
2. Anti-tower penalty preventing 1-word-per-line vertical stacks.
3. Pure-Python Indonesian syllable hyphenator for long words (KBBI rules).
4. Line-count and font-size co-optimization for authentic scanlation feel.
"""

from __future__ import annotations

import math
import re
from typing import Sequence
from PIL import ImageDraw, ImageFont


# ==============================================================================
# 1. Indonesian Syllabification (KBBI Rules, Zero External Dependencies)
# ==============================================================================

class IndonesianHyphenator:
    """Pure-Python syllable splitter for Indonesian comic dialogue."""

    VOWELS = set("aeiouAEIOU")
    DIGRAPHS = ("ng", "ny", "sy", "kh", "NG", "NY", "SY", "KH")
    SUFFIXES = ("kan", "nya", "lah", "kah", "pun", "mu", "ku")

    @classmethod
    def is_vowel(cls, char: str) -> bool:
        return char in cls.VOWELS

    @classmethod
    def _split_base_syllables(cls, core: str) -> list[str]:
        """Split a root word using standard phonetic rules."""
        if len(core) <= 3:
            return [core]

        # Tokenize digraphs as single phonetic units
        tokens: list[str] = []
        i = 0
        while i < len(core):
            if i + 1 < len(core) and core[i:i + 2].lower() in ("ng", "ny", "sy", "kh"):
                tokens.append(core[i:i + 2])
                i += 2
            else:
                tokens.append(core[i])
                i += 1

        types = ["V" if cls.is_vowel(t[0]) else "C" for t in tokens]
        n = len(tokens)
        breaks = [False] * (n + 1)

        idx = 0
        while idx < n:
            # V-V pattern (split between vowels unless diphthong)
            if idx + 1 < n and types[idx] == "V" and types[idx + 1] == "V":
                diphthong = (tokens[idx] + tokens[idx + 1]).lower()
                if diphthong not in ("ai", "au", "oi", "ei"):
                    breaks[idx + 1] = True
            # V-C-V pattern (split before consonant)
            elif idx + 2 < n and types[idx] == "V" and types[idx + 1] == "C" and types[idx + 2] == "V":
                breaks[idx + 1] = True
            # V-C-C-V pattern (split between consonants)
            elif idx + 3 < n and types[idx] == "V" and types[idx + 1] == "C" and types[idx + 2] == "C" and types[idx + 3] == "V":
                breaks[idx + 2] = True
            # V-C-C-C-V pattern (split between 1st and 2nd consonant)
            elif idx + 4 < n and types[idx] == "V" and types[idx + 1] == "C" and types[idx + 2] == "C" and types[idx + 3] == "C" and types[idx + 4] == "V":
                breaks[idx + 2] = True
            idx += 1

        syllables: list[str] = []
        cur: list[str] = []
        for j in range(n):
            if breaks[j] and cur:
                syllables.append("".join(cur))
                cur = []
            cur.append(tokens[j])
        if cur:
            syllables.append("".join(cur))

        return syllables or [core]

    @classmethod
    def hyphenate_word(cls, word: str) -> list[str]:
        """Split an Indonesian word into its syllables with affix awareness."""
        m_lead = re.match(r"^[^\w]+", word)
        lead = m_lead.group(0) if m_lead else ""
        m_trail = re.search(r"[^\w]+$", word)
        trail = m_trail.group(0) if m_trail else ""

        core = word[len(lead):len(word) - len(trail)] if trail else word[len(lead):]
        if len(core) < 5:
            return [word]

        core_lower = core.lower()
        suffix_part = ""

        # Handle common Indonesian suffixes
        for sfx in cls.SUFFIXES:
            if core_lower.endswith(sfx) and len(core) >= len(sfx) + 4:
                suffix_part = core[-len(sfx):]
                core = core[:-len(sfx)]
                core_lower = core.lower()
                break

        # Handle -an suffix (special check for root words ending with ng + an, e.g. denting-an)
        if not suffix_part and core_lower.endswith("an") and len(core) >= 6:
            if core_lower.endswith("ngan"):
                # denting + an
                suffix_part = core[-2:]  # "an"
                core = core[:-2]        # "...ng"
            elif not cls.is_vowel(core[-3]):
                suffix_part = core[-2:]
                core = core[:-2]

        syllables = cls._split_base_syllables(core)
        if suffix_part:
            syllables.append(suffix_part)

        if not syllables:
            return [word]

        syllables[0] = lead + syllables[0]
        syllables[-1] = syllables[-1] + trail
        return syllables

    @classmethod
    def get_hyphenation_candidates(cls, word: str) -> list[tuple[str, str]]:
        """Return valid (prefix-, suffix) break points for a word."""
        clean = re.sub(r"[^\w]", "", word)
        if len(clean) < 8:
            return []

        syllables = cls.hyphenate_word(word)
        if len(syllables) <= 1:
            return []

        candidates = []
        for i in range(1, len(syllables)):
            prefix = "".join(syllables[:i])
            suffix = "".join(syllables[i:])
            if len(re.sub(r"[^\w]", "", prefix)) >= 3 and len(re.sub(r"[^\w]", "", suffix)) >= 3:
                candidates.append((prefix + "-", suffix))
        return candidates


# ==============================================================================
# 2. Ellipse Line Budgets (Chord Width Calculation)
# ==============================================================================

def compute_ellipse_line_budgets(
    bubble_w: int,
    bubble_h: int,
    line_count: int,
    line_height: int,
    spacing: int,
    margin_ratio: float = 0.85,
) -> list[int] | None:
    """Calculate the max readable horizontal width for each line inside an ellipse."""
    total_text_h = line_count * line_height + max(0, line_count - 1) * spacing
    if total_text_h > bubble_h * 0.94:
        return None

    a = (bubble_w / 2.0) * margin_ratio
    b = (bubble_h / 2.0) * margin_ratio

    if a <= 10 or b <= 10:
        return None

    budgets: list[int] = []
    for i in range(line_count):
        line_top = -total_text_h / 2.0 + i * (line_height + spacing)
        line_mid = line_top + line_height / 2.0

        y_ratio = line_mid / b
        if abs(y_ratio) >= 0.98:
            return None

        chord = 2.0 * a * math.sqrt(max(0.0, 1.0 - y_ratio * y_ratio))
        budgets.append(max(15, int(math.floor(chord))))

    return budgets


# ==============================================================================
# 3. Diamond & Balanced Wrapping with Syllable Break Support
# ==============================================================================

def wrap_words_to_budgets(
    words: list[str],
    font: ImageFont.FreeTypeFont,
    budgets: list[int],
    allow_hyphenation: bool = True,
) -> list[str] | None:
    """Partition words across lines to satisfy budget widths with diamond preference."""
    k = len(budgets)
    n = len(words)
    if k <= 0 or n <= 0:
        return None
    if n < k:
        return None

    word_lens = [font.getlength(w) for w in words]
    space_len = font.getlength(" ")

    memo: dict[tuple[int, int], list[str] | None] = {}

    def search(w_idx: int, l_idx: int) -> list[str] | None:
        if (w_idx, l_idx) in memo:
            return memo[(w_idx, l_idx)]

        if l_idx == k:
            return [] if w_idx == n else None
        if w_idx == n:
            return None

        budget = budgets[l_idx]
        best_solution: list[str] | None = None
        best_score = float("-inf")

        cur_text = words[w_idx]
        cur_len = word_lens[w_idx]

        # Single word exceeds budget -> check if hyphenation helps
        if cur_len > budget:
            if allow_hyphenation and len(words[w_idx]) >= 8:
                for prefix, suffix in IndonesianHyphenator.get_hyphenation_candidates(words[w_idx]):
                    if font.getlength(prefix) <= budget:
                        sub = search_with_suffix(suffix, w_idx + 1, l_idx + 1)
                        if sub is not None:
                            cand = [prefix] + sub
                            score = evaluate_partition_quality(cand, budgets, font)
                            if score > best_score:
                                best_score = score
                                best_solution = cand
            memo[(w_idx, l_idx)] = best_solution
            return best_solution

        for next_w in range(w_idx + 1, n + 1):
            if cur_len > budget:
                break

            remaining_words = n - next_w
            remaining_lines = k - (l_idx + 1)
            if remaining_words < remaining_lines:
                break

            sub = search(next_w, l_idx + 1)
            if sub is not None:
                cand = [cur_text] + sub
                score = evaluate_partition_quality(cand, budgets, font)
                if score > best_score:
                    best_score = score
                    best_solution = cand

            if next_w < n:
                cur_text = f"{cur_text} {words[next_w]}"
                cur_len += space_len + word_lens[next_w]

        memo[(w_idx, l_idx)] = best_solution
        return best_solution

    def search_with_suffix(suffix: str, w_idx: int, l_idx: int) -> list[str] | None:
        if l_idx >= k:
            return None
        s_len = font.getlength(suffix)
        budget = budgets[l_idx]
        if s_len <= budget:
            sub = search(w_idx, l_idx + 1)
            if sub is not None:
                return [suffix] + sub
        return None

    return search(0, 0)


def evaluate_partition_quality(
    lines: list[str],
    budgets: list[int],
    font: ImageFont.FreeTypeFont,
) -> float:
    """Score a line partition: penalizes vertical towers, rewards diamond silhouette."""
    score = 100.0
    k = len(lines)
    if k == 0:
        return -9999.0

    line_lens = [font.getlength(line) for line in lines]
    word_counts = [len(line.split()) for line in lines]
    total_words = sum(word_counts)

    # 1. Anti-Tower Guard
    single_word_lines = sum(1 for count in word_counts if count == 1)
    if k >= 3:
        if single_word_lines == k:
            score -= 250.0  # Entire block is 1-word-per-line (Tower)
        elif single_word_lines > k / 2:
            score -= 100.0  # More than half the lines are single words
        else:
            # Check individual 1-word lines: only penalize short orphan words
            for line, count in zip(lines, word_counts):
                if count == 1:
                    clean_w = re.sub(r"[^\w]", "", line)
                    if len(clean_w) <= 4:
                        score -= 35.0  # Very short word alone on a line
                    elif len(clean_w) <= 6:
                        score -= 10.0
    elif k == 2 and single_word_lines == 2 and total_words >= 3:
        score -= 80.0

    # 2. Reward diamond profile for 3+ lines
    if k >= 3:
        mid = k // 2
        # Middle line should ideally be the widest
        if line_lens[mid] >= line_lens[0] and line_lens[mid] >= line_lens[-1]:
            score += 35.0
        if line_lens[0] < line_lens[mid] * 0.96:
            score += 10.0
        if line_lens[-1] < line_lens[mid] * 0.96:
            score += 10.0

    # 3. Penalize orphan prepositions/conjunctions at end of line
    for line in lines[:-1]:
        last_word = line.split()[-1].upper()
        if last_word in ("DI", "KE", "DAN", "YANG", "DARI", "PADA", "UNTUK"):
            score -= 25.0

    # 4. Budget fit ratio
    for length, budget in zip(line_lens, budgets):
        if length > budget:
            score -= 500.0
        else:
            fill = length / max(1.0, budget)
            if 0.50 <= fill <= 0.95:
                score += 15.0 * fill

    return score


# ==============================================================================
# 4. Global Layout Evaluation & Font Search
# ==============================================================================

def target_line_count_range(word_count: int, char_count: int = 0) -> tuple[int, int]:
    """Determine ideal minimum and maximum line count based on word and character counts."""
    if word_count <= 2:
        return 1, 1
    if word_count <= 4:
        return 1, 2
    if word_count <= 6:
        # If words are unusually long (e.g. 36+ chars), allow 4 lines
        return 2, 4 if char_count >= 36 else 3
    if word_count <= 10:
        return 2, 4
    if word_count <= 16:
        return 3, 5
    return 4, 7


def fit_text_v4(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    width: int,
    height: int,
    minimum: int,
    maximum: int,
    spacing_ratio: float,
    stroke_width: int = 0,
    is_curved_balloon: bool = True,
) -> dict | None:
    """Find the most aesthetically balanced, readable typesetting layout.

    Co-optimizes font size and line count using diamond line budgets.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return None

    words = cleaned.split()
    word_count = len(words)
    min_lines, max_lines = target_line_count_range(word_count, len(cleaned))

    best_candidate: dict | None = None
    best_overall_score = float("-inf")

    # Search from preferred size down to minimum size
    for size in range(maximum, minimum - 1, -1):
        try:
            font = ImageFont.truetype(font_path, size=size)
        except OSError:
            continue

        spacing = max(1, round(size * spacing_ratio))
        line_height = size

        # Try viable line counts for this size
        for lines_count in range(min_lines, max_lines + 1):
            if is_curved_balloon:
                budgets = compute_ellipse_line_budgets(
                    width, height, lines_count, line_height, spacing, margin_ratio=0.85
                )
            else:
                budgets = [max(10, int(width * 0.88))] * lines_count

            if not budgets:
                continue

            wrapped_lines = wrap_words_to_budgets(words, font, budgets, allow_hyphenation=True)
            if not wrapped_lines:
                continue

            wrapped_text = "\n".join(wrapped_lines)
            bbox = draw.multiline_textbbox(
                (0, 0), wrapped_text, font=font, spacing=spacing, align="center",
                stroke_width=stroke_width,
            )
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            if text_w > width or text_h > height:
                continue

            partition_score = evaluate_partition_quality(wrapped_lines, budgets, font)
            # High weight on font size so large readable fonts are prioritized over tiny text
            font_size_bonus = (size / max(1, maximum)) * 120.0
            line_overflow_pen = max(0, lines_count - min_lines) * 6.0

            total_score = partition_score + font_size_bonus - line_overflow_pen

            if total_score > best_overall_score:
                best_overall_score = total_score
                best_candidate = {
                    "font": font,
                    "font_size": size,
                    "wrapped": wrapped_text,
                    "spacing": spacing,
                    "text_bbox": bbox,
                    "line_count": len(wrapped_lines),
                    "score": total_score,
                }

        # Early exit if we find a very high-quality fit at >= 80% maximum size
        if best_candidate and best_candidate["score"] >= 200.0 and best_candidate["font_size"] >= round(maximum * 0.80):
            break

    return best_candidate
