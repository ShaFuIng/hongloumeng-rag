from pathlib import Path
import zipfile,json
from rag_core import load_model,build_database,Retriever

p=Path(__file__).resolve().parent
model=load_model()
manifest=build_database((p/'hongloumeng.txt').read_text(encoding='utf-8-sig'),p/'database',model)
with zipfile.ZipFile(p/'rag-data.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in (p/'database').iterdir():z.write(f,f.name)
    z.write(p/'characters.png','characters.png')
r=Retriever(p/'database',model)
tests=[('晴雯撕扇時寶玉有什麼反應',31),('黛玉葬花',27),('探春抄檢大觀園',74),('黛玉焚稿',97),('劉姥姥吃飯逗大家笑',40)]
results=[]
for q,expected in tests:
    hits=r.search(q,6)
    results.append({'question':q,'expected_chapter':expected,'retrieved_chapters':[d['chapter'] for d in hits],'hit':any(d['chapter']==expected for d in hits),'evidence':hits})
report={'manifest':manifest,'tests':results,'llm_tested':False}
(p/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'count':manifest['count'],'tests':[{k:v for k,v in x.items() if k!='evidence'} for x in results]},ensure_ascii=False))
