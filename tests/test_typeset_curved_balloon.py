from PIL import Image, ImageDraw

from typeset_test import fit_text, sanitize_typeset_text


def test_translation_sanitizer_removes_missing_glyph_boxes():
    raw = "HEI JO YURI\u200b\u25a1, KAU DI MANA\uff1f \u201cCEPAT\u201d\u2026"
    cleaned = sanitize_typeset_text(raw)
    assert cleaned == 'HEI JO YURI, KAU DI MANA? "CEPAT"...'
    assert "\u200b" not in cleaned
    assert "\u25a1" not in cleaned


def test_long_translation_shrinks_below_preferred_font(tmp_path):
    # Use the project's production dialogue face: this regression specifically
    # guards against accepting the preferred 42 px size in a narrow balloon.
    font = "assets/fonts/wildwords/CC Wild Words Roman.ttf"
    draw = ImageDraw.Draw(Image.new("RGB", (400, 300), "white"))
    fitted = fit_text(
        draw,
        "A-AKU NGGAK TAHU CARA MENGHINDAR.",
        font,
        width=190,
        height=125,
        minimum=14,
        maximum=42,
        spacing_ratio=0.14,
        stroke_width=0,
    )
    assert fitted is not None
    assert fitted["font_size"] < 42
    assert fitted["text_bbox"][2] - fitted["text_bbox"][0] <= 190
    assert fitted["text_bbox"][3] - fitted["text_bbox"][1] <= 125
