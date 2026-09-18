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

def terminology_errors(items,rows,glossary):
    errors=[]
    for source,row in zip(items,rows):
        occupied=[]
        for term,target in sorted(glossary.items(),key=lambda pair:len(pair[0]),reverse=True):
            if not term or not target:continue
            matches=[m for m in term_pattern(term,True).finditer(source['text']) if not any(m.start()<b and m.end()>a for a,b in occupied)]
            if not matches:continue
            occupied.extend(m.span() for m in matches)
            if not term_pattern(target).search(row['translation']):errors.append(f"{source['id']}: retain {term!r} as {target!r}")
    return errors

def grounded_updates(updates,items,glossary):
    sources='\n'.join(item['text'] for item in items)
    existing={key.casefold() for key in glossary}
    return {key:value for key,value in updates.items() if key.casefold() not in existing and term_pattern(key,True).search(sources)}
