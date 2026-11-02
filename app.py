import os, json, re, html
from pathlib import Path
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
import gradio as gr
import aisuite as ai
from rag_core import Retriever, build_messages, CHAPTERS

LLM_MODEL='groq:openai/gpt-oss-120b'
BASE=Path(__file__).resolve().parent
retriever=None
client=None

def initialize(folder, api_key=None):
    global retriever,client
    retriever=Retriever(folder)
    if api_key: os.environ['GROQ_API_KEY']=api_key
    client=ai.Client() if os.environ.get('GROQ_API_KEY') else None

def lookup(question,history,k,chapter,mode):
    if not question.strip(): return history,'請先輸入問題。', '', question
    history=list(history or [])
    search_text=question
    if len(question)<35 and any(x in question for x in ['他','她','後來','為什麼','然後']):
        previous=[m['content'] for m in history if m['role']=='user']
        if previous: search_text=previous[-1]+' '+question
    try: evidence=retriever.search(search_text,int(k),None if chapter=='全部收錄回目' else int(chapter))
    except ValueError as exc:return history,str(exc),'',question
    context='\n\n'.join(f"### [{d['id']}] 第{d['chapter']}回\n{d['text']}" for d in evidence)
    status=''
    if mode=='只查原文（不呼叫模型）':
        answer='以下是相關原文片段，尚未經大語言模型整理。\n\n'+context
        status='原文檢索模式；不是生成式回答。'
    elif client is None:
        return history,'尚未設定 Groq 金鑰。請執行 Colab 金鑰步驟，或切換「只查原文」。',context,question
    else:
        try:
            response=client.chat.completions.create(model=LLM_MODEL,messages=build_messages(question,evidence,history),temperature=0.15,max_tokens=1800)
            answer=response.choices[0].message.content or ''
            if not answer.strip():raise ValueError('empty')
            allowed={d['id'] for d in evidence}
            cited=set(re.findall(r'\[(C\d{3}-\d{5})\]',answer))
            if cited-allowed:status='回答含未提供的片段編號，請核對右側原文。'
            elif not cited:status='本次回答沒有原文編號；請核對原文是否足以回答。'
            else:status='引用編號已核對；答案是否受原文支持仍請對照閱讀。'
            if getattr(response.choices[0],'finish_reason',None)=='length':status+=' 回答達長度上限，可能未完整。'
        except Exception:
            return history,'模型呼叫失敗：請檢查 Groq 金鑰、模型權限或額度後重試。',context,question
    history.extend([{'role':'user','content':question},{'role':'assistant','content':answer}])
    return history,status,context,''

CSS='''
.gradio-container {max-width:1180px!important;background:#faf7f0!important;color:#394c43!important;
--body-text-color:#394c43!important;--body-text-color-subdued:#68736c!important;
--background-fill-primary:#faf7f0!important;--background-fill-secondary:#f1ece2!important;
--block-background-fill:#fffdf8!important;--block-title-text-color:#394c43!important;
--block-label-background-fill:#eee8db!important;--block-label-text-color:#394c43!important;
--input-background-fill:#fffdf8!important;--input-text-color:#394c43!important;}
.gradio-container .prose,.gradio-container .prose * {color:#394c43!important;}
#hero {position:relative;overflow:hidden;border-radius:22px;border:1px solid #e4d8c8;background:#f6f0e5;min-height:300px;margin-bottom:20px;}
#hero img {width:100%;height:340px;object-fit:cover;object-position:center 43%;display:block;}
#hero .copy {position:absolute;left:4%;top:22%;width:34%;color:#3f514b;}
#hero h1 {font-family:serif;font-size:42px;line-height:1.3;margin:12px 0;color:#3f514b!important;}
#hero p {font-size:15px;line-height:1.8;color:#3f514b!important;}
.small-label {font-size:12px;letter-spacing:3px;color:#896755;}
footer {display:none!important;}
#evidence {max-height:530px;overflow-y:auto;}
@media(max-width:650px){#hero img{height:270px;opacity:.65}#hero .copy{width:70%;top:12%}#hero h1{font-size:30px}}
'''

def make_demo():
    import base64
    banner=BASE/'characters.png'
    image='data:image/png;base64,'+base64.b64encode(banner.read_bytes()).decode() if banner.exists() else ''
    with gr.Blocks(title='紅樓夢・劇情查詢') as demo:
        gr.HTML('<div id="hero">'+(f'<img src="{image}" alt="寶玉、黛玉與寶釵的Q版插畫">' if image else '')+'<div class="copy"><div class="small-label">人物 · 情節 · 原文</div><h1>紅樓夢<br>劇情查詢</h1><p>記得一段情節，卻忘了誰在做什麼？<br>問一句，回到原文找答案。</p></div></div>')
        gr.Markdown('**原文節選知識庫** · 收錄 '+ '、'.join(map(str,CHAPTERS))+' 回。第 81–120 回內容標記為本版本後40回。')
        with gr.Row():
            with gr.Column(scale=3):
                chat=gr.Chatbot(label='劇情問答',height=450)
                question=gr.Textbox(label='想查哪一段？',placeholder='例如：晴雯撕扇時，寶玉有什麼反應？',lines=2,max_lines=5)
                with gr.Row():
                    ask=gr.Button('查詢情節',variant='primary')
                    clear=gr.Button('清除對話')
                gr.Examples(examples=[['黛玉葬花時，寶玉在做什麼？'],['晴雯為什麼撕扇子？寶玉有生氣嗎？'],['劉姥姥在大觀園吃飯時，怎麼逗大家笑？'],['抄檢大觀園時，探春有什麼反應？'],['黛玉焚稿時，紫鵑和雪雁在做什麼？'],['寶玉的手機是什麼牌子？']],inputs=question)
            with gr.Column(scale=2):
                gr.Markdown('### 原文查證\n回答中的片段編號，可在下方找到。相似度只用於檢索排序，不代表答案正確率。')
                status=gr.Textbox(label='查詢狀態',interactive=False)
                sources=gr.Markdown(value='提出問題後，這裡會顯示實際送給模型的原文。',sanitize_html=True,elem_id='evidence')
                gr.Markdown('[文本來源：Project Gutenberg #24264](https://www.gutenberg.org/ebooks/24264)')
        with gr.Accordion('查詢設定',open=False):
            mode=gr.Radio(['RAG 回答','只查原文（不呼叫模型）'],value='RAG 回答',label='模式')
            chapter=gr.Dropdown(['全部收錄回目']+[str(n) for n in CHAPTERS],value='全部收錄回目',label='限定回目')
            k=gr.Slider(3,10,value=6,step=1,label='檢索片段數')
        gr.Markdown('閱讀助理以平實繁體中文回答，不扮演角色。圖片為 AI 生成的裝飾插畫，不作為劇情證據。')
        inputs=[question,chat,k,chapter,mode];outputs=[chat,status,sources,question]
        ask.click(lookup,inputs,outputs,concurrency_limit=1)
        question.submit(lookup,inputs,outputs,concurrency_limit=1)
        clear.click(lambda:([],'','',''),outputs=[chat,status,sources,question],queue=False)
    return demo

if __name__=='__main__':
    initialize(BASE/'database')
    make_demo().queue(max_size=8).launch(css=CSS,theme=gr.themes.Soft(primary_hue='emerald',neutral_hue='stone'),share=False)
