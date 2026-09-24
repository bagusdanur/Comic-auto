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
Do not force trendy slang into historical, fantasy, serious or formal scenes.
Names, places and named skills follow canonical terminology. Ordinary jobs and
objects can use familiar Indonesian. Approved terminology wins over style.
Read interrupted bubbles as a sequence; preserve unfinished speech, emotion,
jokes and plot facts. Never invent a speaker identity or fill missing context.
Silently edit every line for naturalness, ambiguity and terminology before
returning JSON. Shorten only without losing meaning. Output no editorial notes.'''

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
