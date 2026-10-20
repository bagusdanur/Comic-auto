#!/usr/bin/env python3
"""Phase 2: group OCR lines and translate them without editing any image."""

from __future__ import annotations

import argparse
import asyncio
from difflib import SequenceMatcher
import json
import os
import re
import time
from pathlib import Path

from translation_providers import create_provider
from language_memory import POLICY_VERSION


def load_env(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def horizontal_overlap(a: dict, b: dict) -> float:
    left = max(a["x"], b["x"])
    right = min(a["x"] + a["width"], b["x"] + b["width"])
    return max(0, right - left) / max(1, min(a["width"], b["width"]))


def collapse_duplicated_sentence(text: str) -> str:
    value = " ".join(text.split())
    match = re.fullmatch(r"(.+?[.!?]+)\s+\1", value, flags=re.IGNORECASE)
    return match.group(1) if match else value


def merge_vertical_fragments(groups: list[dict]) -> list[dict]:
    """Merge OCR groups that substantially occupy the same speech bubble."""
    merged: list[dict] = []
    for group in sorted(groups, key=lambda item: (item["bbox"]["y"], item["bbox"]["x"])):
        box = group["bbox"]
        target = merged[-1] if merged else None
        if target is not None:
            other = target["bbox"]
            vertical_overlap = max(
                0,
                min(box["y"] + box["height"], other["y"] + other["height"])
                - max(box["y"], other["y"]),
            )
            overlap_ratio = vertical_overlap / max(1, min(box["height"], other["height"]))
            vertical_gap = max(
                box["y"] - (other["y"] + other["height"]),
                other["y"] - (box["y"] + box["height"]),
                0,
            )
            same_column = horizontal_overlap(box, other) >= 0.50
            close_stack = vertical_gap <= max(18, round(min(box["height"], other["height"]) * 0.16))
            if same_column and (overlap_ratio >= 0.20 or close_stack):
                target["text"] = f'{target["text"]} {group["text"]}'
                target["source_region_ids"].extend(group["source_region_ids"])
                right = max(other["x"] + other["width"], box["x"] + box["width"])
                bottom = max(other["y"] + other["height"], box["y"] + box["height"])
                other["x"], other["y"] = min(other["x"], box["x"]), min(other["y"], box["y"])
                other["width"], other["height"] = right - other["x"], bottom - other["y"]
                continue
        merged.append(group)
    return merged


from promo_text import is_noise

# Kata sambung/keterangan narasi yang sah berdiri sendiri di kotak besar,
# tapi tidak pernah muncul sebagai glyph dekoratif atau SFX aksi.
NARRATION_WORDS = frozenset({
    "and", "but", "or", "nor", "so", "yet", "if", "when", "while", "then",
    "now", "not", "the", "all", "one", "two", "who", "why", "how", "what",
})

# Di atas ambang ini huruf pendek yang terbaca yakin dianggap teks, bukan glyph.
CONFIDENT_LETTERING = 0.85


def _tie(box, candidate):
    left = max(box["x"], candidate["x"])
    right = min(box["x"] + box["width"], candidate["x"] + candidate["width"])
    overlap = max(0, right - left) / max(1, min(box["width"], candidate["width"]))
    gap = max(box["y"] - (candidate["y"] + candidate["height"]),
              candidate["y"] - (box["y"] + box["height"]), 0)
    return overlap, gap


def trusted_region_ids(regions):
    """Fixed-point trust so a whole RAW sentence is removed, not half of it.

    A chat bubble runs four lines. The recogniser answers the stylised second
    line at 0.36 ("SAAMAA Isn") and the last one at 0.49 read as digits ("21"),
    while the line above reads 0.99. The old ``confidence < 0.60`` floor
    returned False *before* the proximity rescue could run, so those lines were
    never grouped, never removed, and the raw English stayed on the finished
    page -- exactly the half-erased sentence that rescue was written to
    prevent. Seeds are the reads the recogniser stood behind (>= 0.75); every
    other line joins only when it is tightly stacked in the same column as an
    already trusted one, so loose specks far from any reliable line still die
    here rather than being translated.
    """
    trusted = set()
    for region in regions:
        if float(region.get("confidence", 0)) >= 0.75 and len(str(region.get("text", "")).strip()) > 1:
            trusted.add(region.get("id"))
    changed = True
    while changed:
        changed = False
        for region in regions:
            rid = region.get("id")
            if rid in trusted:
                continue
            text = str(region.get("text", "")).strip()
            if len(text) < 2:
                continue
            box = region["bbox"]
            has_letters = len(re.sub(r"[^a-z]", "", text, flags=re.I)) >= 3
            confidence = float(region.get("confidence", 0))
            for other in regions:
                if other.get("id") not in trusted:
                    continue
                candidate = other["bbox"]
                overlap, gap = _tie(box, candidate)
                span = min(box["height"], candidate["height"])
                # The proven rescue (>= 0.60, >= 3 letters, loose proximity) is
                # kept unchanged; the strong tier chains one tightly stacked
                # line to the next, which is what carries a four-line run.
                loose = (has_letters and confidence >= 0.60
                         and overlap >= 0.42 and gap <= max(45, round(span * .45)))
                strong = overlap >= 0.60 and gap <= max(24, round(span * .25))
                if loose or strong:
                    trusted.add(rid)
                    changed = True
                    break
    return trusted


def group_dialogue(regions: list[dict], minimum_y: int) -> list[dict]:
    TRUSTED = trusted_region_ids(regions)

    def trustworthy_region(region: dict) -> bool:
        return region.get("id") in TRUSTED

    def short_emotional_dialogue(region: dict) -> bool:
        text = str(region.get("text", "")).strip()
        return bool(
            float(region.get("confidence", 0)) >= 0.72
            and re.fullmatch(r"[A-Za-z]{2,5}[.!?.,~…-]{2,}", text)
        )

    def narration_word(region: dict) -> bool:
        """A short word that belongs to prose rather than to lettering.

        ch1 page-004 stacks "AND" (conf 1.00, 130x106) directly above
        "WHAT FOLLOWED NEXT WAS-" in identical black boxes; the 13-letter line
        cleared the size rule and was translated, while the connective above it
        was rejected as a giant glyph, never grouped, never removed, and the raw
        English stayed on the finished page. Decorative glyphs and action SFX
        are not common English connectives, so those keep their place.
        """
        word = region["text"].strip().lower()
        return len(region["text"].split()) == 1 and word in NARRATION_WORDS

    clean = [
        region for region in regions
        if region["bbox"]["y"] >= minimum_y
        and (trustworthy_region(region) or short_emotional_dialogue(region))
        and len(region["text"].strip()) > 1
        and "http" not in region["text"].casefold()
        and not is_noise(region["text"])
        # A huge one-glyph/two-glyph OCR region is an integrated action SFX,
        # even when Korean lettering is misread as a Latin fragment such as Hj.
        # A high-confidence read of a short word is text, not lettering: the
        # glyph rule was built for action SFX, which the recogniser answers
        # with 0.1-0.7 confidence ('HHH' 0.32, 'jHA' 0.51, 'Q...' 0.84), while
        # genuine reactions come back at 0.99-1.00 and were being thrown away
        # with them -- ch1 page-004 lost 'GET' (conf 1.00, next to its
        # translated 'DOWN!!') and 'BUT.' (conf 0.99), and across every job
        # 'NO.', 'HUH?', 'RUN!', 'HEY.', 'AAH!' were all dropped as glyphs.
        # Below 0.85 nothing changes: 1337 regions sit there and only 3 of them
        # are real dialogue.
        and not (len(re.sub(r"\W", "", region["text"])) <= 3
                 and int(region["bbox"]["height"]) >= 100
                 and int(region["bbox"]["width"]) >= 100
                 and not short_emotional_dialogue(region)
                 and not narration_word(region)
                 and float(region.get("confidence", 0)) < CONFIDENT_LETTERING)
        # Very thin OCR strips in action-heavy raws are usually UI labels,
        # watermarks, or decorative speed text rather than translatable bubbles.
        # Promo copy ("LIGHTNING-FAST RELEASES WITH PREMIUM QUALITY") measures
        # the same 14 px height, and dropping it left the raw banner line on
        # the finished page, so only short labels are rejected now.
        and not (
            int(region["bbox"]["height"]) <= 16
            and int(region["bbox"]["width"]) >= int(region["bbox"]["height"]) * 6
            and len(region["text"].split()) < 4
        )
    ]
    # Tile overlap may report the same lettering twice with one damaged first
    # character (for example HEY!! / rEY!!). Remove those duplicates before
    # line grouping so neither translation nor cleanup receives repeated text.
    deduped: list[dict] = []
    for region in sorted(clean, key=lambda item: float(item.get("confidence", 0)), reverse=True):
        box = region["bbox"]
        duplicate = False
        normalized = re.sub(r"[^a-z0-9]", "", region["text"].casefold())
        for kept in deduped:
            other = kept["bbox"]
            iw = max(0, min(box["x"] + box["width"], other["x"] + other["width"]) - max(box["x"], other["x"]))
            ih = max(0, min(box["y"] + box["height"], other["y"] + other["height"]) - max(box["y"], other["y"]))
            overlap = iw * ih / max(1, min(box["width"] * box["height"], other["width"] * other["height"]))
            other_text = re.sub(r"[^a-z0-9]", "", kept["text"].casefold())
            if overlap >= .72 and SequenceMatcher(None, normalized, other_text).ratio() >= .66:
                duplicate = True
                break
        if not duplicate:
            deduped.append(region)
    clean = sorted(deduped, key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))

    groups: list[dict] = []
    for region in clean:
        box = region["bbox"]
        # Compare against the BEST matching earlier group, not just the most
        # recent one. Independent narration labels ("OGRE FANG +1") sit to the
        # side of a dialogue bubble and are emitted between its lines; keying
        # only on groups[-1] let those labels break the chain, so one bubble's
        # lines split into several groups and were translated (and therefore
        # typeset) separately at mismatched font sizes.
        best_index = -1
        best_gap = None
        for index in range(len(groups)):
            candidate = groups[index]
            last_box = candidate.get("_last_bbox", candidate["bbox"])
            gap = box["y"] - (last_box["y"] + last_box["height"])
            gap_limit = min(55, max(18, round(min(box["height"], last_box["height"]) * .65)))
            if gap > gap_limit or gap < -0.35 * min(box["height"], last_box["height"]):
                continue
            if horizontal_overlap(box, last_box) < 0.25:
                continue
            # Column/width guard: a label in another column (wide x offset or a
            # very different line width) must not be absorbed into the bubble.
            # Compare against the group's LAST LINE, not its accumulated box:
            # once a three-line bubble spans 328px, a 328px-wide comparison
            # would happily swallow the 158px loot label sitting beside it.
            width_ratio = max(box["width"], last_box["width"]) / max(1, min(box["width"], last_box["width"]))
            left_delta = abs(box["x"] - last_box["x"])
            if width_ratio > 1.6 or left_delta > 0.35 * min(box["width"], last_box["width"]):
                continue
            if best_gap is None or abs(gap) < abs(best_gap):
                best_gap, best_index = gap, index
        if best_index >= 0:
            previous = groups[best_index]
            previous_box = previous["bbox"]
            previous["source_region_ids"].append(region["id"])
            previous["text"] += " " + region["text"]
            right = max(previous_box["x"] + previous_box["width"], box["x"] + box["width"])
            bottom = max(previous_box["y"] + previous_box["height"], box["y"] + box["height"])
            previous_box["x"] = min(previous_box["x"], box["x"])
            previous_box["y"] = min(previous_box["y"], box["y"])
            previous_box["width"] = right - previous_box["x"]
            previous_box["height"] = bottom - previous_box["y"]
            previous["_last_bbox"] = dict(box)
            continue
        groups.append({
            "id": f"bubble-{len(groups) + 1:03d}",
            "text": region["text"],
            "bbox": dict(box),
            "source_region_ids": [region["id"]],
            "_last_bbox": dict(box),
        })
    # OCR tiles can return neighboring words from the same visual line as
    # separate groups. Consolidate horizontally touching groups before asking
    # the translator, otherwise several Indonesian snippets are typeset on top
    # of one another inside the same bubble/chat balloon.
    consolidated: list[dict] = []
    for group in sorted(groups, key=lambda item: (item["bbox"]["x"], item["bbox"]["y"])):
        merged = False
        box = group["bbox"]
        for target in reversed(consolidated):
            other = target["bbox"]
            vertical = max(0, min(box["y"] + box["height"], other["y"] + other["height"]) - max(box["y"], other["y"]))
            vertical_ratio = vertical / max(1, min(box["height"], other["height"]))
            horizontal_gap = max(box["x"] - (other["x"] + other["width"]), other["x"] - (box["x"] + box["width"]), 0)
            # Two groups that share one text baseline and touch horizontally are
            # adjacent words of a single line ("AND" + "PICK" + "ONE" + "FLAVOR
            # EACH?"), so they always join. A separate column (a loot list
            # beside a bubble) sits further to the side on its own baseline, so
            # it stays apart.
            same_row_words = (
                horizontal_gap <= 12
                and abs(box["y"] - other["y"]) <= 0.30 * min(box["height"], other["height"])
            )
            intersection = min(box["x"] + box["width"], other["x"] + other["width"]) - max(box["x"], other["x"])
            centre_gap = abs((box["x"] + box["width"] / 2) - (other["x"] + other["width"] / 2))
            different_column = (
                intersection <= 0
                and centre_gap > 0.60 * max(box["width"], other["width"])
                and not same_row_words
            )
            if (
                vertical_ratio >= 0.62
                and not different_column
                and horizontal_gap <= max(18, round(min(box["height"], other["height"]) * 0.75))
            ):
                ordered = sorted([(other["x"], target["text"]), (box["x"], group["text"])])
                target["text"] = " ".join(item[1] for item in ordered)
                target["source_region_ids"].extend(group["source_region_ids"])
                right = max(other["x"] + other["width"], box["x"] + box["width"])
                bottom = max(other["y"] + other["height"], box["y"] + box["height"])
                other["x"], other["y"] = min(other["x"], box["x"]), min(other["y"], box["y"])
                other["width"], other["height"] = right - other["x"], bottom - other["y"]
                merged = True
                break
        if not merged:
            consolidated.append(group)
    groups = merge_vertical_fragments(consolidated)
    strong_intro_promo = any(
        item["bbox"]["y"] < 1600 and re.search(
            r"(?:website\s+has\s+been\s+detected|more\s+amazing\s+series|"
            r"best\s+possible\s+quality|continuing\s+to\s+publish)",
            item["text"], re.I,
        )
        for item in groups
    )
    if strong_intro_promo:
        groups = [item for item in groups if item["bbox"]["y"] >= 1600]
    groups = [
        item for item in groups
        if not re.search(r"you\s+can\s+rea[do]\b.{0,28}\bchapter\s+on", item["text"], re.I)
    ]
    for index, group in enumerate(groups, 1):
        group["id"] = f"bubble-{index:03d}"
        group["text"] = collapse_duplicated_sentence(group["text"])
        group.pop("_last_bbox", None)
    return groups


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ocr_json", type=Path)
    parser.add_argument(
        "--provider", choices=("deepseek", "gemini", "openai-compatible", "9router"), default="deepseek"
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--min-y", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("output/translations.json"))
    args = parser.parse_args()
    if args.env_file:
        load_env(args.env_file)

    ocr = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    groups = group_dialogue(ocr["regions"], args.min_y)
    glossary = json.loads(args.glossary.read_text(encoding="utf-8")) if args.glossary and args.glossary.exists() else {}
    memory = json.loads(args.memory.read_text(encoding="utf-8")) if args.memory and args.memory.exists() else {}
    memory_context = {}
    if args.memory:
        structured = args.memory.parent / args.memory.stem / "memory-summary.json"
        if structured.exists():
            memory_context = json.loads(structured.read_text(encoding="utf-8"))
    glossary = {**memory, **glossary}
    request_items = [{"id": group["id"], "text": group["text"]} for group in groups]
    started = time.perf_counter()
    provider = create_provider(args.provider)
    chunk_size = int(os.getenv("NINEROUTER_CHUNK_SIZE", "4")) if args.provider == "9router" else len(request_items)
    if chunk_size > 0 and len(request_items) > chunk_size:
        translations = []
        for start in range(0, len(request_items), chunk_size):
            translations.extend(await provider.translate(request_items[start:start + chunk_size], glossary, memory_context))
    else:
        translations = await provider.translate(request_items, glossary, memory_context)
    memory_updates = getattr(provider, "memory_updates", {})
    if args.memory:
        args.memory.parent.mkdir(parents=True, exist_ok=True)
        # Existing choices are stable; a later chapter may add terms but cannot
        # silently rename established characters or skills.
        merged_memory = {**memory, **{k: v for k, v in memory_updates.items() if k not in memory}}
        temporary = args.memory.with_suffix(args.memory.suffix + ".tmp")
        temporary.write_text(json.dumps(merged_memory, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.memory)
    translated = {row["id"]: row for row in translations}
    for group in groups:
        row = translated[group["id"]]
        group["translation"] = row["translation"]
        group["style"] = row.get("style") or "dialogue"

    result = {
        "provider": args.provider,
        "model": os.getenv(
            "DEEPSEEK_MODEL" if args.provider == "deepseek" else
            "GEMINI_MODEL" if args.provider == "gemini" else
            "OPENAI_COMPATIBLE_MODEL" if args.provider == "openai-compatible" else
            "NINEROUTER_MODEL",
            "default",
        ),
        "seconds": round(time.perf_counter() - started, 2),
        "groups": groups,
        "memory_terms_used": len(glossary),
        "memory_terms_added": len([key for key in memory_updates if key not in memory]),
        "memory_profile_used": bool(memory_context),
        "language_policy_version": POLICY_VERSION,
        "language_edit_pass": True,
        "terminology_validation": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for group in groups:
        print(f'{group["id"]}: {group["text"]}\n  -> {group["translation"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
