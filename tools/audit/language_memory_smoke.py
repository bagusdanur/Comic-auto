"""Read-only live translation evaluation using latest Chapter 5 sources."""
import asyncio,json
from pathlib import Path
from translate_text_test import load_env
from translation_providers import create_provider
from language_memory import POLICY_VERSION

async def main():
    load_env(Path('.env.9router'))
    base=Path('data/jobs/library/erica-when-a-friend-is-obsessed-with-me/chapter-5/44e06345-7314-48b4-b369-8a300f21b23d/metadata')
    memory=json.loads(Path('data/memory/erica-when-a-friend-is-obsessed-with-me.json').read_text())
    memory.update({'S-Class Mercenary':'tentara bayaran kelas S','Merchant Guild':'serikat pedagang'})
    items=[]
    for page,numbers in [(3,[4]),(5,[5,8,9,10]),(7,[2,6,13]),(9,[1,6])]:
        for g in json.loads((base/f'page-{page:03d}'/'translations.json').read_text())['groups']:
            items.append({'id':f"p{page}-{g['id']}",'text':g['text']})
    provider=create_provider('9router')
    rows=await provider.translate(items,memory)
    report={'policy':POLICY_VERSION,'model':provider.model,'pairs':[{'source':a['text'],**b} for a,b in zip(items,rows)]}
    Path('artifacts/language-memory-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
asyncio.run(main())
