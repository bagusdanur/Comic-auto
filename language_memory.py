"""Indonesian editorial policy and conservative terminology validation."""
import re
POLICY_VERSION='natural-id-v2'
LANGUAGE_POLICY='''Use clear spoken Indonesian, not literal English syntax or ornate prose.
Preserve actor, target, negation, comparison direction, cause and consequence.
"These guys don't even compare to wyverns" means they are much weaker:
"Mereka jauh lebih lemah daripada wyvern", not a reversed comparison.
Understand idioms from neighboring dialogue. "Reach out" in contact information
means "hubungi", not "raih". Examples are contextual, never blanket replacements.
"Ignorant" about someone's experiences means not knowing/understanding: use
"belum tahu" or "tidak paham" as context permits, not "picik" (narrow-minded)
unless the source actually expresses narrow-mindedness. Preserve degree naturally.
An occupational address such as "Master Merchant!" can be "Pak!" when it is an
ordinary respectful address; do not invent a royal title or flatten a formal rank.
Keep each speaker's established voice; do not switch aku/kamu to gue/lo or
saya/Anda without contextual reason. Politeness depends on the relationship,
not simply age. Avoid stiff filler, redundant phrases and unnecessary English.
Keep ONE register inside a single sentence: never mix formal saya/Anda with
colloquial nggak/bikin/udah, and never mix literary bukankah/sekiranya with
colloquial bikin. Formal scene -> "TERSERAH ANDA! SAYA TIDAK PEDULI!",
"Bukannya Anda sendiri yang membuat kegaduhan?"; casual scene -> "Terserah
kamu! Aku nggak peduli!", "Bukannya kamu sendiri yang bikin ribut?".
Do not force trendy slang into historical, fantasy, serious or formal scenes.
Names, places and named skills follow canonical terminology. Ordinary jobs and
objects can use familiar Indonesian. Approved terminology wins over style.
Read interrupted bubbles as a sequence; preserve unfinished speech, emotion,
jokes and plot facts. Never invent a speaker identity or fill missing context.
Silently edit every line for naturalness, ambiguity and terminology before
returning JSON. Shorten only without losing meaning. Output no editorial notes.'''

def restore_aliases(text, aliases):
    """Kembalikan varian istilah/nama ke bentuk kanonik dari memori.

    `aliases` memetakan bentuk yang salah (biasanya hasil terjemahan model)
    ke bentuk kanonik, mis. {"Penyihir Es": "Ice Witch", "Tori": "TOURI"}.
    Hanya token utuh yang diganti supaya kata lain tidak ikut berubah.
    """
    if not aliases or not text:
        return text
    for variant, canonical in sorted(aliases.items(), key=lambda pair: len(pair[0]), reverse=True):
        if not variant or not canonical:
            continue
        if str(variant).casefold() == str(canonical).casefold():
            continue
        text = term_pattern(str(variant)).sub(str(canonical), text)
    return text


def term_pattern(term,plural=False):
    return re.compile(r'(?<!\w)'+re.escape(term)+(r'(?:s)?' if plural and term[-1:].isalpha() else '')+r'(?!\w)',re.I)

# Ordinary English function words that are never a character name. An entry
# like "An" -> "An" (seen when a memory run mistook the article for a name)
# makes terminology_errors demand the literal token in every translation
# starting with "An ...", which no correct Indonesian sentence can satisfy;
# the job then fails after its retries. These are skipped on both the read
# and the write side so a bad memory entry can neither block a chapter nor be
# created by a later one.
STOPWORD_TERMS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'if', 'in', 'on', 'at', 'to', 'of',
    'is', 'are', 'was', 'were', 'be', 'do', 'does', 'did', 'not', 'no', 'yes',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that',
    'these', 'those', 'as', 'by', 'for', 'with', 'so', 'than', 'then', 'there',
}


def enforce_term(term, target):
    """True when a glossary pair is safe to require at validation time."""
    key = str(term or '').strip()
    value = str(target or '').strip()
    if not key or not value:
        return False
    # A stopword can never be a required proper noun.
    if key.casefold() in STOPWORD_TERMS:
        return False
    # Case-insensitive identity of a single short word ("An" -> "An", "EL" ->
    # "EL") carries no translation instruction, so requiring it only risks
    # false failures: the Indonesian line may legitimately drop the token.
    if key.casefold() == value.casefold() and len(key) <= 3 and key.isalpha():
        return False
    return True


def terminology_errors(items,rows,glossary):
    errors=[]
    for source,row in zip(items,rows):
        occupied=[]
        for term,target in sorted(glossary.items(),key=lambda pair:len(pair[0]),reverse=True):
            if not enforce_term(term,target):continue
            matches=[m for m in term_pattern(term,True).finditer(source['text']) if not any(m.start()<b and m.end()>a for a,b in occupied)]
            if not matches:continue
            occupied.extend(m.span() for m in matches)
            if not term_pattern(target).search(row['translation']):errors.append(f"{source['id']}: retain {term!r} as {target!r}")
    return errors

def grounded_updates(updates,items,glossary):
    sources='\n'.join(item['text'] for item in items)
    existing={key.casefold() for key in glossary}
    return {key:value for key,value in updates.items()
            if key.casefold() not in existing
            and enforce_term(key,value)
            and term_pattern(key,True).search(sources)}

# Pemanggilan (honorifik) yang sering dipilih acak oleh model: Pak/Mas/Tuan
# untuk sumber yang sama ("Mr. Takeru"). Semua varian sebelum nama yang sama
# dinormalkan ke satu bentuk kanonik agar tidak berubah antar halaman.
ADDRESS_TOKENS = (
    "Pak", "Bapak", "Mas", "Mbak", "Tuan", "Nona", "Nyonya", "Bu", "Kak",
    "Om", "Tante", "Sdr", "Saudara", "Mr", "Mr.", "Sir", "Madam", "Young Master",
)

GENERIC_ADDRESS = {
    "mr": "Tuan", "mr.": "Tuan", "mister": "Tuan", "sir": "Tuan",
    "young master": "Tuan Muda", "miss": "Nona", "mrs": "Nyonya", "mrs.": "Nyonya",
}

# Gender honorifik Indonesia. Dipakai untuk memverifikasi bahwa sapaan yang
# dipelajari cocok dengan gender di teks sumber, bukan sekadar kemunculan
# pertama hasil model (yang dulu mengunci "Bu Guru" untuk semua guru).
MALE_HONORIFICS = {"pak", "bapak", "mas", "tuan", "tuan muda", "om"}
FEMALE_HONORIFICS = {"bu", "ibu", "mbak", "nona", "nyonya", "tante"}

# Sinyal gender yang andal dari teks sumber Inggris. Kata ambigu seperti
# "miss" (bisa kata kerja) sengaja tidak dipakai supaya tidak salah kunci.
_MALE_SOURCE = re.compile(
    r"\b(he|him|his|himself|mr\.?|mister|sir|young master)\b", re.I
)
_FEMALE_SOURCE = re.compile(
    r"\b(she|her|hers|herself|mrs\.?|madam|young lady)\b", re.I
)


def honorific_gender(honorific: str) -> str:
    value = str(honorific or "").strip().casefold()
    if value in MALE_HONORIFICS:
        return "male"
    if value in FEMALE_HONORIFICS:
        return "female"
    return "neutral"


def source_gender(text: str) -> str | None:
    """Gender dari kata ganti/gelar sumber. None bila tidak ada bukti atau campur."""
    value = str(text or "")
    male = bool(_MALE_SOURCE.search(value))
    female = bool(_FEMALE_SOURCE.search(value))
    if male and not female:
        return "male"
    if female and not male:
        return "female"
    return None

_HONORIF_ALT = "|".join(re.escape(token) for token in ADDRESS_TOKENS)


def _address_overrides(policy: dict | None) -> dict[str, str]:
    """Peta `address` dari editorial-policy.json (kunci: frasa sumber)."""
    if not isinstance(policy, dict):
        return {}
    value = policy.get("address")
    if not isinstance(value, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip() and str(v).strip()}


def _rewrite(text: str, name: str, canonical: str) -> str:
    """Ganti semua bentuk panggilan untuk satu nama dengan bentuk kanonik."""
    pattern = re.compile(r"\b(" + _HONORIF_ALT + r"|" + re.escape(name) + r"-(?:san|sama|kun|chan))\b[ .-]*"
                         r"(" + re.escape(name) + r")\b", re.IGNORECASE)

    def replace(match):
        # Huruf besar/kecil mengikuti honorifik aslinya supaya lettering kapital
        # tidak berubah jadi campuran.
        word = canonical.upper() if match.group(1).isupper() else canonical
        return f"{word} {match.group(2)}"

    text = pattern.sub(replace, text)
    # bentuk "Name-san" tanpa kata lain
    text = re.sub(r"\b" + re.escape(name) + r"-(?:san|sama|kun|chan)\b",
                  f"{canonical} {name}", text, flags=re.IGNORECASE)
    return text


def _canonical_honorific(canonical: str, name: str) -> str:
    """`Pak Takeru` + nama `Takeru` -> `Pak` supaya tidak dobel."""
    if canonical.casefold().endswith(name.casefold()):
        trimmed = canonical[: -len(name)].strip(" .-")
        return trimmed or canonical
    return canonical


def normalize_address_terms(translation: str, source: str, policy: dict | None = None) -> str:
    """Samakan honorifik untuk satu nama mengikuti bentuk kanonik.

    Contoh: sumber `Mr. Takeru`, hasil `Mas Takeru` / `Tuan Takeru` /
    `Takeru-san` semuanya menjadi `Pak Takeru` (atau sesuai peta judul).
    Hanya berjalan kalau sumber memang memakai honorifik untuk nama itu.
    """
    if not translation or not source:
        return translation
    overrides = _address_overrides(policy)
    text = translation
    # Peta berbasis nama: {"Takeru": "Pak Takeru"} hasil pembelajaran otomatis
    # dari chapter sebelumnya (tanpa perlu tahu frasa sumbernya).
    if isinstance(policy, dict):
        for name, canonical in (policy.get("address_names") or {}).items():
            name, canonical = str(name).strip(), str(canonical).strip()
            if name and canonical:
                text = _rewrite(text, name, _canonical_honorific(canonical, name))
    for phrase, canonical in overrides.items():
        if phrase.casefold() not in source.casefold():
            continue
        name = phrase.split()[-1]
        text = _rewrite(text, name, _canonical_honorific(canonical, name))
    def _key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    for match in re.finditer(r"\b(Mr\.?|Mister|Sir|Young Master|Miss|Mrs\.?)\s+([A-Z][A-Za-z'\-]+)", source):
        title = match.group(1).lower()
        # Gelar/kata sifat ("Mr. Guru", "Miss Muda") bukan nama diri; jangan
        # dipaksa jadi honorifik gender ke seluruh judul.
        from chapter_context import NON_NAME_TOKENS
        if match.group(2).casefold() in NON_NAME_TOKENS:
            continue
        # Lewati kalau sapaan+nama ini sudah diatur peta judul atau hasil belajar.
        learned = {_key(name) for name in (policy.get("address_names") or {})} if isinstance(policy, dict) else set()
        if _key(f"{match.group(1)} {match.group(2)}") in {_key(phrase) for phrase in overrides}:
            continue
        if _key(match.group(2)) in learned:
            continue
        canonical = GENERIC_ADDRESS.get(title) or GENERIC_ADDRESS.get(title.rstrip("."))
        if not canonical:
            continue
        text = _rewrite(text, match.group(2), canonical)
    return text


def enforce_established_address(text: str, established: list[str] | None) -> str:
    """Pakai sapaan yang sudah muncul di halaman sebelumnya.

    Konteks chapter menyimpan entri seperti "Pak Takeru"; kalau halaman ini
    menulis "Mas Takeru"/"Tuan Takeru", bentuk itu disamakan tanpa perlu peta
    per judul.
    """
    if not text or not established:
        return text
    for entry in established:
        parts = str(entry).strip().split()
        if len(parts) < 2:
            continue
        canonical, name = " ".join(parts[:-1]), parts[-1]
        text = _rewrite(text, name, canonical)
    return text
