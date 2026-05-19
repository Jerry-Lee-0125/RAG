
""" 
1. 請將 preprocess.py 檔案放在同一路徑下 
2. Word 轉檔的部分需要先安裝 LibreOffice: https://zh-tw.libreoffice.org/download/
3. 安裝完 LibreOffice 需設定環境變數: 
   搜尋程式「環境變數」 -> 點擊右下角「環境變數(N)」 -> 找到 Path 或 path 的變數點選「編輯」
   -> 點擊右上角「新增」 -> 貼上 C:\Program Files\LibreOffice\program -> 確定
"""

import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from preprocess import sync_and_convert_files # 引入前處理函式(比對檔案修改時間、轉檔、清理刪除檔案)

# ==========================================
# 1. 初始化 LLM 與 嵌入模型 (Embeddings)
# ==========================================
llm = OllamaLLM(
    #model='weitsung50110/llama-3-taiwan:8b-instruct-dpo-q8_0',
    model='gemma3:4b', # 最高支援到 128k Tokens
    num_ctx=8192,    # 如未設定，Ollama預設是 2048 Tokens
    temperature=0,   # 降低隨機性，減少幻覺
    callbacks=[StreamingStdOutCallbackHandler()]
)

embeddings = OllamaEmbeddings(
    model='bge-m3'
)

# ==========================================
# 2. 定義路徑與前處理(文件分類與轉檔)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 取得目前所在路徑

raw_data_path = os.path.join(BASE_DIR, "01_raw_data") # 原始病理科文件資料夾路徑
"""會在目前路徑下建立以下資料夾："""
processed_data_path = os.path.join(BASE_DIR, "02_processed_data") # 處理後的文件(讀取用)
faiss_path = os.path.join(BASE_DIR, "03_db", "faiss_index_1") # 向量資料庫儲存路徑
error_data_path = os.path.join(BASE_DIR, "04_error_data") # 轉檔失敗的檔案隔離路徑


"""
同步處理 01_raw_data 裡被 "新增"、"修改"、"刪除" 的檔案到 02_processed_data
如是舊版 doc 檔，也會同步轉檔成 docx 檔，並放到 02_processed_data 中
如是 xls、xlsx 檔，會轉檔成 csv 檔，並放到 02_processed_data 中
"""
has_changes = sync_and_convert_files(raw_data_path, processed_data_path, error_data_path) 

# ==========================================
# 3. 文本分割與向量資料庫建立
# ==========================================
if has_changes or not os.path.exists(faiss_path):
    print("偵測到檔案有變更，或尚未建立資料庫，開始(重新)載入文件並建立向量資料庫...")
    # 1. 載入 PDF 檔案
    print("正在載入 PDF 檔案...")
    pdf_loader = DirectoryLoader(
        processed_data_path, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader, 
        show_progress=True 
    )
    pdf_docs = pdf_loader.load()
    
    # 2. 載入 DOCX 檔案
    print("正在載入 Word 檔案...")
    docx_loader = DirectoryLoader(
        processed_data_path, 
        glob="**/*.docx", 
        loader_cls=Docx2txtLoader, 
        show_progress=True 
    )
    docx_docs = docx_loader.load()

    # 3. 載入 CSV 檔案 
    csv_loader = DirectoryLoader(
        processed_data_path, 
        glob="**/*.csv", 
        loader_cls=CSVLoader, 
        loader_kwargs={'encoding': 'utf-8-sig'}, # utf-8-sig，避免中文變成亂碼
        show_progress=True 
        )
    csv_docs = csv_loader.load()

    # 4. 合併所有載入的文件
    docs = pdf_docs + docx_docs + csv_docs
    print(f"已載入所有文件，共有 {len(docs)} 個頁面")
    
    # 文本分割器設置 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=125,  
        separators=["\n\n", "\n", "。", "，", " ", ""]  
    )

    # 分割文件 
    split_docs = text_splitter.split_documents(docs)
    print(f"總共拆分了 {len(split_docs)} 個 chunk")

    # 建立並儲存向量資料庫 
    vectordb = FAISS.from_documents(split_docs, embeddings)
    vectordb.save_local(faiss_path)
    print("向量資料庫建立完畢並已成功儲存！")
else:
    print("已從本地加載現有的向量資料庫！") #檔案未變動且資料庫已存在，直接讀取向量資料庫
    vectordb = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)


# ==========================================
# 4. 建立檢索器與問答鏈
# ==========================================
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.4} # 設定檢索門檻
    )

system_prompt = (
    "你是一位專業的醫院病理科助手。你的任務是根據所提供的參考資料（Context）來回答問題。\n\n"
    "回答規範：\n"
    "1. 僅根據提供的參考資料進行回答。如果資料中沒有提到相關資訊，請誠實回答「抱歉，在目前的病理科規範資料中找不到相關資訊」，不可隨意編造。\n"
    "2. 若資料提及多項流程或步驟，請使用條列式清單呈現。\n"
    "3. 嚴禁根據你自身的訓練知識來補充資料庫以外的醫療建議或行政流程。\n"
    "4. 所有的回答必須使用繁體中文，並維持專業、嚴謹且親切的語氣。\n\n"
    "參考資料如下：\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('user', '問題: {input}')
])

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(
    retriever, document_chain
)


# ==========================================
# 5. 主程式：處理用戶輸入與檢索
# ==========================================
if __name__ == "__main__":
    print("歡迎使用病理科助手！")
    while True:
        print("輸入您的問題(輸入 'bye' 結束程式):")
        user_query = input('>>> ')
        if user_query.lower() in ['bye', '退出', '結束']:
            print("感謝您的使用，再見！")
            break

        # 呼叫 retrieval_chain 處理用戶問題，進行檢索與回答
        response = retrieval_chain.invoke({'input': user_query})
        
        print("\n")
        print("\n[檢索資料來源]:")
        for i, doc in enumerate(response['context']):
            content = doc.page_content.strip()
            print(f"來源 {i+1}: {doc.metadata.get('source', '未知')}") # 取得來源路徑
            #print(f"內容: {content}") # 取得檢索片段內容
            print("\n")
        print("=" * 50)
        print("\n")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        