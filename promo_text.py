"""Promotional / source-marking copy detection shared by every stage.

``translate_text_test`` rejects this copy before grouping (it is never
translated) and ``remove_text_test`` needs the identical verdict so the
rejected line is still erased. Keeping one definition here guarantees the
two stages can never drift apart. Import this module from anywhere: it
pulls in nothing but :mod:`re`.
"""
import re


def is_noise(text: str) -> bool:
    """True for promotional/source-marking copy that is never translated.

    Hoisted out of ``group_dialogue`` because the removal stage needs exactly
    the same verdict: a region this function rejects never reaches a group, so
    nobody would ever erase it and the raw promo line survived into the final
    page. Keep both callers on one definition so they can never drift apart.
    """
    value = text.strip().casefold()
    compact = re.sub(r"[^a-z0-9]", "", value)
    return (
        not re.search(r"[a-z]", value)
        or bool(re.search(
            r"(?:https?://|www\.|\.org\b|\.com\b|discord|paypal\.me|"
            r"make\s+sure\s+to\s+read|only\s+at\s+our\s+website|"
            r"read\s+this\s+series\s+only|read\s+only\s+at|"
            r"read\s+(?:at|on)\b|read\b.{0,24}\b(?:chapter|series)\b.{0,16}\b(?:at|on)|"
            r"lightning[-\s]*fast\s+releases?|premium\s+quality\s+(?:thriving\s+)?community|"
            r"discover\s+more\s+chapters?|"
            r"chapters?\s+up\s+to\s+\d+|now\s+available\s+exclusively|"
            r"follow\s+for|faster\s+updates?|large\s+library|"
            r"thunder\s*scans?|oi\s*scans?|(?:please\s+)?support\s+(?:us|our)|support\s+means|"
            r"(?:we\s+are|we're)\s+recruiting|(?:korean|japanese)\s+translators?|"
            r"proofreaders?|quality\s+checkers?|\beditors?\s*:\s*\d|"
            r"proofreaders?\b.{0,80}\b(?:translators?|editors?))", value
        ))
        or bool(re.search(
            r"(?:reader\b.{0,36}\b(?:support|website)|more\s+amazing\s+series|"
            r"best\s+possible\s+quality|continuing\s+to\s+publish)", value
        ))
        or any(token in compact for token in (
            "evascans", "readthisseriesonly", "scanlation",
            "makesuretoreadthisonlyatourwebsite", "enthunderscans",
            "thunderscansteam", "qiscans", "oiscans",
        ))
    )
