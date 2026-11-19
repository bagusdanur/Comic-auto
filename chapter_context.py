"""Konteks lintas halaman untuk satu chapter.

Setiap halaman diterjemahkan oleh proses terpisah, jadi tanpa berkas ini model
tidak tahu sapaan/istilah yang sudah dipakai halaman sebelumnya ("Mr. Takeru"
bisa jadi Pak/Mas/Tuan di halaman berbeda). Berkas ini menyimpan beberapa baris
terakhir plus istilah/sapaan yang sudah mapan, lalu dikirim ulang ke setiap
permintaan terjemahan berikutnya.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MAX_LINES = 60
MAX_PAGES = 4
ADDRESS_PATTERN = re.compile(
    r"\b(Pak|Bapak|Mas|Mbak|Tuan Muda|Tuan|Nona|Nyonya|Bu|Kak|Om|Tante)\s+([A-Z][A-Za-z'\-]+)\b"
)

# Gelar, kata sifat, dan kata sapaan lain BUKAN nama diri. Regex di atas
# menangkap apa pun berhuruf kapital setelah honorifik, sehingga "Bu Guru",
# "Tuan Muda", dan "Pak Tua" dulu terkunci sebagai nama "Guru"/"Muda"/"Tua".
# Akibatnya semua guru dipaksa jadi Bu (termasuk guru laki-laki) dan semua
# "Tuan Muda" jadi Nona di judul lain. Token-token ini tidak boleh dipelajari
# sebagai nama, supaya gender tidak pernah dikunci dari gelar/kata sifat.
NON_NAME_TOKENS = {
    # gelar / profesi
    "guru", "dokter", "dok", "kapten", "komandan", "jenderal", "profesor", "prof",
    "ustad", "ustadz", "ustazah", "kyai", "haji", "hajjah", "pendeta", "pastor",
    "ratu", "raja", "pangeran", "putri", "duke", "duchess", "lord", "lady",
    "teacher", "doctor", "master", "mistress", "sir", "madam", "miss", "mr", "mrs",
    "captain", "commander", "general", "professor", "king", "queen", "prince", "princess",
    # kata sifat / penunjuk yang sering menempel pada honorifik
    "muda", "tua", "besar", "kecil", "agung", "mulia", "yang", "itu", "ini",
    "young", "old", "great", "little", "elder", "eldest",
    # sapaan kekerabatan (bukan nama diri)
    "kakak", "adik", "paman", "bibi", "nenek", "kakek", "ibu", "bapak", "ayah", "bunda",
    "papa", "mama", "kak", "om", "tante", "mas", "mbak", "pak", "bu", "tuan", "nona",
    "nyonya", "sdr", "saudara", "saudari", "brother", "sister", "uncle", "aunt",
    "grandma", "grandpa", "father", "mother", "dad", "mom",
}


def iter_address(text: str):
    """Yield (honorifik, nama) hanya untuk nama diri sungguhan.

    Gelar/kata sifat ("Guru", "Muda", "Tua") dilewati supaya tidak pernah
    dikunci sebagai nama dan memaksa gender yang salah ke seluruh judul.
    """
    for match in ADDRESS_PATTERN.finditer(str(text or "")):
        name = match.group(2)
        if name.casefold() in NON_NAME_TOKENS:
            continue
        yield match.group(1), name


def load(path: str | Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"lines": [], "address": [], "pages": 0}
    if not isinstance(data, dict):
        return {"lines": [], "address": [], "pages": 0}
    data.setdefault("lines", [])
    data.setdefault("address", [])
    data.setdefault("pages", 0)
    return data


def prompt_block(context: dict) -> dict | None:
    """Potongan ringkas untuk ditempel ke memory_context terjemahan."""
    lines = [str(line) for line in context.get("lines", []) if str(line).strip()]
    address = [str(item) for item in context.get("address", []) if str(item).strip()]
    if not lines and not address:
        return None
    return {
        "instruction": (
            "Ini dialog halaman sebelumnya di chapter yang sama. Pertahankan sapaan, "
            "istilah, nama, dan register yang sudah dipakai; jangan mengganti bentuk "
            "panggilan yang sudah mapan."
        ),
        "address_used": address[-12:],
        "previous_lines": lines[-MAX_LINES:],
    }


def update(path: str | Path, page_number: int, groups: list[dict]) -> dict:
    """Tambahkan hasil satu halaman ke konteks chapter."""
    context = load(path)
    new_lines = []
    for group in groups:
        text = str(group.get("translation") or "").strip()
        if text:
            new_lines.append(text)
        for honorific, name in iter_address(text):
            entry = f"{honorific} {name}"
            if entry not in context["address"]:
                context["address"].append(entry)
    if new_lines:
        context["lines"].extend(new_lines)
    context["lines"] = context["lines"][-MAX_LINES * MAX_PAGES :]
    context["pages"] = max(int(context.get("pages") or 0), page_number)
    Path(path).write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return context
