# 紅樓夢・劇情查詢

用日常問題查詢人物、事件與回目，並顯示實際原文依據。採用《紅樓夢》20個完整回目節選，644個原文片段。

## 開啟作業

- [RAG02：在 Colab 開啟對話機器人](https://colab.research.google.com/github/ShaFuIng/hongloumeng-rag/blob/main/RAG02.ipynb)
- [RAG01：在 Colab 建立向量資料庫](https://colab.research.google.com/github/ShaFuIng/hongloumeng-rag/blob/main/RAG01.ipynb)
- [直接下載 rag-data.zip](https://raw.githubusercontent.com/ShaFuIng/hongloumeng-rag/main/rag-data.zip)

RAG02 已填好上方 ZIP 網址，可直接依序執行，不需先執行 RAG01。

## 執行方式

1. 開啟 RAG02 的 Colab 連結。建議使用 T4 GPU。
2. Colab 左側鑰匙圖示新增名為 `Groq` 的密鑰，開啟筆記本存取權。請勿把密鑰寫進程式、截圖或公開儲存庫。
3. 依序執行儲存格，啟動 Gradio。
4. 使用「RAG 回答」模式查詢；右側會顯示實際提供給模型的原文。

未設定密鑰時可切換「只查原文」，但此模式不是大語言模型生成回答。Colab 執行需要登入 Google；閱讀這份公開筆記本不需 GitHub 權限。

## 資料與設計

原文來自 [Project Gutenberg #24264](https://www.gutenberg.org/ebooks/24264)，完整下載原檔為 `hongloumeng.txt`，來源授權文字保留在原檔與 ZIP 的 `SOURCE-LICENSE.txt`。

收錄第3、6、23、26、27、28、31、33、34、37、38、40、41、73、74、96、97、98、119、120回。此為特定電子文本節選，非完整120回知識庫；後40回內容另外標示。

RAG01：原文切段與重疊 → multilingual-e5-small 向量化 → FAISS → rag-data.zip。

RAG02：下載ZIP與校驗 → FAISS語意檢索＋關鍵詞排序 → AIsuite／Groq生成 → Gradio展示。

角色為平實的繁體中文閱讀助理，直接回答「誰在做什麼」。Prompt要求檢查錯誤前提、附片段ID與回目；證據不足時明確說明，不將未檢索到當作全書不存在。

## 驗證狀態

已完成資料庫建立、ZIP校驗、Notebook程式語法檢查、五組檢索測試及Gradio原文查詢操作。五組測試皆在前六筆結果找到預期回目，詳細結果見 `validation.json`。此結果不代表所有問題都正確。

**尚未使用Groq密鑰完成生成回答測試，也尚未在全新Colab環境完成端到端測試。** `原文檢索實測截圖.png` 是原文檢索模式，不能當成生成回答截圖。正式繳交前須完成這兩項驗證並補上截圖。

## AI協助揭露

程式、Prompt、介面及人物插畫由Codex與圖片生成工具協助製作，知識來源為小說原文。繳交者須自行閱讀理解、驗證答案，依課程要求附上AI提示詞與生成結果截圖，並改寫為自己的理解。

## 老師範例

- https://yenlung.me/rag01colab
- https://yenlung.me/rag02colab

保留老師指定套件及版本，另自訂資料切分、檢索、Prompt與介面。建庫與查詢使用相同的E5向量模型及revision。

## 介面預覽

![Gradio介面](介面完整截圖.png)
