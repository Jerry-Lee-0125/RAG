"""
版本更新日誌：
2026-06-16 v4.0

- 1.新增功能：系統管理員模式
    網頁左上角新增登入功能，輸入密碼後才能進入系統管理員模式，進行檔案管理(新增/刪除)與資料庫重建等敏感操作，防止未授權使用者誤觸或惡意操作，提升系統安全性。

- 2.新增功能：檔案摘要預覽
    在系統管理員模式的「檔案管理」視窗中新增「檔案摘要預覽」分頁，使用者可以點選目前資料庫內的任一檔案，AI 助手會即時生成該檔案內容的精簡摘要，幫助使用者快速了解文件重點，提升使用體驗。

- 3.移除功能：LLM 溫度設定控制
    為了確保生成內容的專業性與一致性，將 LLM 的溫度設定固定為 0.0，移除使用者調整溫度的選項，避免因為誤觸導致回答出現不準確的資訊。

- 4.新增防呆機制:檢索結果為空時直接阻斷 LLM 生成
    在 RAG 檢索階段加入防呆機制，如果檢索到的相關文本為空，系統會直接阻斷後續的 LLM 生成過程，避免 LLM 在沒有足夠上下文的情況下產生不準確或無意義的回答，提升整體的使用體驗與可靠性。

- 5.新增版面設計：免責聲明固定於頁面底部
    「⚕️ 病理科助手為 AI 輔助系統，有時可能產生錯誤資訊。結果僅供參考，請務必與專業醫療規範確認。」

- 6.新增模型：gpt-oss:20b
    在伺服器上看到有這個模型，應該是另一組下載的模型，我也有在模型選擇新增這個選項，可以試試看這個模型，目前測試有比 llama-3-taiwan:8b 和 gemma3:4b 的回答好很多。

"""

import os
import re
import math
import glob
import shutil
import subprocess
import streamlit as st
import pandas as pd
from datetime import datetime
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 設定網頁標題
st.set_page_config(page_title="病理科檢索問答系統", layout="wide")

# ==========================================
# 1. 初始化 LLM 與 嵌入模型
# ==========================================
# 載入大型語言模型 (LLM)
@st.cache_resource
def get_llm(model_name, keep_alive_time=None):
    # 建立基礎參數字典
    llm_kwargs = {
        "model": model_name,
        "num_ctx": 8192,      # 如未設定，Ollama預設是 2048 Tokens
        "temperature": 0.0    # 控制溫度(生成隨機性)
    }
    
    # 如果有特別指定 keep_alive 時間，就把它加進參數裡
    if keep_alive_time is not None:
        llm_kwargs["keep_alive"] = keep_alive_time
        
    return OllamaLLM(**llm_kwargs)


# 載入嵌入模型 (Embeddings)
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model='bge-m3')
    
embeddings = get_embeddings()

# ==========================================
# 2. 定義路徑
# ==========================================
# 取得目前程式執行檔所在的絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 存放使用者上傳並轉檔完成的檔案路徑
processed_data_path = os.path.join(BASE_DIR, "01_processed_data") 
# 存放 Chroma 向量資料庫的路徑  
chroma_path = os.path.join(BASE_DIR, "02_db", "chroma_db")       
# 確保資料夾存在，若不存在則系統自動建立
os.makedirs(processed_data_path, exist_ok=True)

# ==========================================
# 3. 向量資料庫操作 (支援局部新增/刪除)
# ==========================================
def get_vector_db():
    """讀取現有的 Chroma 向量資料庫"""
    return Chroma(persist_directory=chroma_path, embedding_function=embeddings)

def add_file_to_db(file_path, vectordb):
    """將單一檔案讀取、切塊並新增至 Chroma 資料庫，
    優點：支援局部更新，不用每次上傳新文件就全部重建資料庫，節省運算資源"""
    ext = os.path.splitext(file_path)[1].lower() # 取得副檔名並轉小寫
    
    # 根據副檔名選擇對應的 LangChain 載入器
    if ext == '.pdf':
        docs = PyPDFLoader(file_path).load()
    elif ext == '.docx':
        docs = Docx2txtLoader(file_path).load()
    elif ext == '.csv':
        docs = CSVLoader(file_path, encoding='utf-8-sig').load() # 病理科表單常包含中文，使用 utf-8-sig 避免中文亂碼
    else:
        return False 
    
    if docs:
        # 資料清洗階段：先將所有載入的文本進行空白壓縮
        for doc in docs:
            raw_text = doc.page_content
            # 1. 將連續的空白行強制壓縮成標準的雙換行
            clean_text = re.sub(r'\n\s*\n+', '\n\n', raw_text)
            # 2. 清除每一行開頭的空白與 Tab，保持對齊
            clean_text = "\n".join([line.lstrip() for line in clean_text.split('\n')])
            # 3. 清除整個段落頭尾多餘的空白與隱藏字元
            doc.page_content = clean_text.strip()
            
        # 進行文本分割：將長文章切成小塊，方便向量檢索比對
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,     
            chunk_overlap=150,  # 重疊 150 字，避免跨段落語意被切斷
            separators=["\n\n", "\n", "。", "，", " ", ""]  # 優先依照段落和標點符號做切割
        )
        split_docs = text_splitter.split_documents(docs) 
        
        if split_docs:
            vectordb.add_documents(split_docs) # 將切好的文字塊寫入資料庫
            return True # 成功寫入資料庫，回傳 True
            
    # 如果讀取出來是空的（例如純圖片的 PDF）
    print(f"⚠️ 略過寫入：檔案 {os.path.basename(file_path)} 無法提取純文字內容。")
    return False # ❌ 無效檔案，回傳 False

def remove_file_from_db(file_path, vectordb):
    """根據檔案路徑，把該檔案在資料庫中對應的所有文字塊 (Chunks) 刪除"""
    try:
        # 透過 Metadata 中的 source (來源路徑) 找到該檔案所有的 Chunk ID
        result = vectordb.get(where={"source": file_path})
        ids_to_delete = result.get("ids", [])
        
        # 若有找到對應的 ID，則執行刪除
        if ids_to_delete:
            vectordb.delete(ids=ids_to_delete)
    except Exception as e:
        print(f"從資料庫移除 {file_path} 時發生錯誤: {e}")

def build_vector_db(vectordb, ui_placeholder=None):
    """強制重整： 遇到資料庫錯亂時，重新掃描 01_processed_data 並重建 Chroma 資料庫"""
    
    log_text = "### 🔄 向量資料庫重建進度\n\n"
    
    def update_ui(msg):
        """用於即時更新網頁畫面的進度提示"""
        nonlocal log_text
        log_text += f"{msg}\n\n"
        if ui_placeholder:
            ui_placeholder.markdown(log_text)
    
    
    try:
        # 清空向量資料庫內的所有資料，而不是刪除實體資料夾，避免遇到「檔案被鎖定」的報錯
        update_ui("✅ **[階段 1/5]** 正在清空舊有向量資料庫...")
        vectordb.delete_collection()
    except Exception as e:
        pass # 如果是空資料庫或初次建立，可能沒有資料可以清，直接略過

    # 重新讀取資料夾內所有支援的檔案
    update_ui("✅ **[階段 2/5]** 正在掃描並讀取硬碟中的病理科檔案...")
    pdf_docs = DirectoryLoader(processed_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    docx_docs = DirectoryLoader(processed_data_path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
    csv_docs = DirectoryLoader(processed_data_path, glob="**/*.csv", loader_cls=CSVLoader, loader_kwargs={'encoding': 'utf-8-sig'}).load()

    docs = pdf_docs + docx_docs + csv_docs

    # 若沒有任何文件，回傳空的 Chroma 實例
    if not docs:
        return get_vector_db()
    
    update_ui("✅ **[階段 3/5]** 正在執行資料清洗...")
    for doc in docs:
        raw_text = doc.page_content
        # 壓縮連續的空白行
        clean_text = re.sub(r'\n\s*\n+', '\n\n', raw_text)
        # 清除每一行開頭的空白與 Tab
        clean_text = "\n".join([line.lstrip() for line in clean_text.split('\n')])
        # 清除頭尾多餘字元
        doc.page_content = clean_text.strip()

    update_ui("✅ **[階段 4/5]** 正在進行文本分割...")    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  
        chunk_overlap=150,  
        separators=["\n\n", "\n", "。", "，", " ", ""]  
    )
    split_docs = text_splitter.split_documents(docs)

    # 建立並儲存新的 Chroma 向量資料庫
    update_ui("✅ **[階段 5/5]** 嵌入模型轉換向量並寫入資料庫...")
    new_vectordb = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings,
        persist_directory=chroma_path 
    )
    return new_vectordb  

# ==========================================
# 4. 檔案管理介面 
# ==========================================
@st.dialog("📁 檔案管理", width="large")
def file_management_center():
    """檔案管理中心：包含上傳新文件與管理現有文件(刪除)的頁籤"""
    tab_upload, tab_manage, tab_preview = st.tabs(["📤 上傳新文件", "🗑️ 管理現有文件", "📄 檔案摘要預覽"])

    # --- 分頁 1：上傳功能 ---
    with tab_upload:
        st.write("支援檔案類型: pdf, docx, doc, csv, xls, xlsx")
        
        # 建立檔案上傳器，支援多檔案同時上傳
        uploaded_files = st.file_uploader(
            "選擇檔案", 
            type=['pdf', 'docx', 'doc', 'csv', 'xls', 'xlsx'], # 限定上傳檔案類型，避免上傳不支援的格式
            accept_multiple_files=True, # 支援多檔案上傳
            key="dialog_uploader"
            )
        
        if st.button("上傳並更新資料庫", icon="💾", use_container_width=True):
            if uploaded_files:
                save_count = 0 # 紀錄成功寫入資料庫的檔案數量
                
                failed_records = [] # 用來收集上傳失敗的檔案清單，方便後續顯示給使用者看
                total_files = len(uploaded_files) # 取得總檔案數
                
                # 在網頁畫面上建立進度條與狀態文字的佔位符
                progress_bar = st.progress(0)
                status_text = st.empty()
                
         
                with st.spinner("系統正在處理檔案，請稍候..."):
                    # 使用 enumerate 取得目前的索引值 (idx)，方便計算進度
                    for idx, file in enumerate(uploaded_files):
                        
                        if file.name.startswith('~'): # 略過系統產生的暫存檔 (例如打開 Word 時產生的 ~$ 檔案)
                            continue
                        
                        # 即時更新網頁上的狀態文字，讓使用者知道目前處理到哪一份文件
                        current_step = idx + 1
                        status_text.write(f"⏳ 正在處理 ({current_step}/{total_files})：**{file.name}** ...")
                        
                        file_path = os.path.join(processed_data_path, file.name)
                        name, ext = os.path.splitext(file.name)
                        ext = ext.lower() # 將副檔名轉為小寫，避免因大小寫差異導致判斷錯誤
                        
                        # 1. --- 攔截加密檔案(防呆機制) ---
                        # 在寫入硬碟前，先檢查檔案是否被密碼保護，避免後續轉檔程式錯誤
                        is_encrypted = False
                        try:
                            import pypdf
                            import msoffcrypto
                            
                            if ext == '.pdf':
                                if pypdf.PdfReader(file).is_encrypted:
                                    is_encrypted = True
                            elif ext in ['.doc', '.docx', '.xls', '.xlsx']:
                                if msoffcrypto.OfficeFile(file).is_encrypted():
                                    is_encrypted = True
                                    
                            # 檢查完畢後，必須將檔案讀取指標歸零 (Seek 0)
                            # 否則下面寫入硬碟時會從檢查結束的地方開始讀，導致存出 0 byte 的損壞檔案
                            file.seek(0) 
                        except Exception:
                            file.seek(0) # 萬一檢查套件失敗，依然歸零放行，交給後面的 except 捕捉

                        if is_encrypted:
                            # 發現加密，加入失敗清單並直接跳過這個檔案，不寫入硬碟
                            failed_records.append({"檔案名稱": file.name, "失敗原因": "檔案已加密，請解除密碼後再上傳"})
                            continue 
              
                
                        # 2. --- 檢查重複檔名 (安全阻擋機制) ---
                        # 避免覆蓋舊檔案導致 Chroma 向量資料庫出現對應不上的資訊
                        # 掃描資料夾內是否有相同主檔名的檔案 (例如找 細胞學檢查規範.*)
                        search_pattern = os.path.join(processed_data_path, f"{name}.*")
                        existing_files = glob.glob(search_pattern)
                        
                        # 如果上傳的是 Excel，也要一併檢查是否已經有之前拆解出來的 CSV 檔
                        search_pattern_csv = os.path.join(processed_data_path, f"{name}_*.csv")
                        existing_files.extend(glob.glob(search_pattern_csv))
                        
                        # 如果找到任何同名的舊檔案，立刻攔阻
                        if existing_files:
                            failed_records.append({
                                "檔案名稱": file.name, 
                                "失敗原因": "已存在同名檔案。為避免資料遺失，請先至「管理現有文件」手動刪除舊檔後再上傳"
                            })
                            continue # 直接跳過這個檔案，不寫入硬碟也不轉檔，繼續處理下一個檔案
                            
                        # 3. --- 正式寫入硬碟 ---
                        # 確定沒有加密與同名衝突後，才將檔案暫存到硬碟中
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        
                        # 4. --- 轉檔與資料庫寫入 ---
                        try:
                            # [狀況 A] 處理舊版 Word (.doc) -> 呼叫本機端 LibreOffice 在背景轉成 .docx 檔
                            if ext == '.doc':
                                soffice_path = shutil.which('libreoffice') or shutil.which('soffice') or r"C:\Program Files\LibreOffice\program\soffice.exe"
                                subprocess.run([soffice_path, '--headless', '--convert-to', 'docx', '--outdir', processed_data_path, file_path], check=True, timeout=60) # --headless 代表不開啟軟體畫面，在背景執行轉檔
                                os.remove(file_path) # 轉檔成功後刪除原始 .doc
                                
                                new_docx_path = os.path.join(processed_data_path, f"{name}.docx")
                                if add_file_to_db(new_docx_path, st.session_state.vectordb):
                                    save_count += 1
                                else:
                                    os.remove(new_docx_path) 
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "轉檔後無法提取純文字"})
                                
                            # [狀況 B] 處理 Excel (.xls, .xlsx) -> 拆解成多個 CSV 工作表
                            # Excel檔中可能帶有多個工作表，透過 pandas 將每個工作表獨立拆分成 CSV，有助於 LLM 精準檢索
                            elif ext in ['.xls', '.xlsx']:
                                excel_dict = pd.read_excel(file_path, sheet_name=None, dtype=str)
                                valid_csv_count = 0
                                for sheet_name, df in excel_dict.items():
                                    # 資料清洗：移除整行或整列都是空值的無效數據
                                    df.dropna(how='all', inplace=True)
                                    df.dropna(how='all', axis=1, inplace=True)
                                    if not df.empty:
                                        csv_filename = f"{name}_{sheet_name}.csv"
                                        csv_path = os.path.join(processed_data_path, csv_filename)
                                        df.to_csv(csv_path, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 編碼，確保繁體中文不會變成亂碼
                                        
                                        if add_file_to_db(csv_path, st.session_state.vectordb):
                                            valid_csv_count += 1
                                        else:
                                            os.remove(csv_path) 
                                            
                                os.remove(file_path) # 拆解完成後刪除原始的 Excel 檔案
                                if valid_csv_count > 0:
                                    save_count += 1
                                else:
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "無法提取有效表格資料"})
                            
                            # [狀況 C] 不需要轉檔的 PDF, DOCX, CSV -> 直接寫入向量資料庫
                            else:
                                if add_file_to_db(file_path, st.session_state.vectordb):
                                    save_count += 1
                                else:
                                    os.remove(file_path) 
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "純圖片或無法解析的內容"})
                                
                        except Exception as e:
                            # 錯誤捕捉：如果轉檔或寫入過程崩潰，將剛剛寫入硬碟的殘留檔案刪除，保持資料夾乾淨
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            error_msg = str(e).lower()
                            # 攔截 Pandas 拋出的加密錯誤，轉換為中文提示
                            if "encrypted" in error_msg or "password" in error_msg:
                                failed_records.append({"檔案名稱": file.name, "失敗原因": "檔案已加密或受密碼保護，請解鎖後再上傳"})
                            else:
                                failed_records.append({"檔案名稱": file.name, "失敗原因": f"系統處理錯誤 ({e})"})
                            
                    # 單個檔案處理完畢後，更新進度條的百分比
                    progress_percentage = current_step / total_files
                    progress_bar.progress(progress_percentage)
                    
                # 迴圈結束後，清除網頁上的進度條與狀態文字，保持版面乾淨
                status_text.empty()
                progress_bar.empty()
                
                
                # 5. --- 上傳結果呈現 ---
                if save_count == len(uploaded_files):
                    # 情況一：全部成功，記錄提示訊息並重新整理網頁
                    st.session_state["show_success_toast"] = f"✅ 成功處理全部 {save_count} 份檔案！"
                    st.rerun()
                else:
                    # 情況二：有部分失敗情況，顯示資料表讓使用者知道哪些檔案有問題
                    if save_count > 0:
                        st.success(f"✅ 已成功寫入 {save_count} 份檔案。")
                    
                    st.error(f"⚠️ 發現 {len(failed_records)} 份檔案無法寫入，上傳失敗，請人工確認內容：")
                    
                    # 利用 Pandas DataFrame 呈現乾淨的表格，hide_index=True 去除最左邊的數字序號
                    df_failed = pd.DataFrame(failed_records)
                    st.dataframe(df_failed, hide_index=True, use_container_width=True)
            else:
                st.error("請先選擇要上傳的檔案。") # 若使用者沒選擇檔案就按下按鈕的防呆提示

    # --- 分頁 2：管理/刪除功能 ---
    with tab_manage:
        file_data = []
        
        # 掃描資料夾內現有的文件，抓取目前所有可檢索的病理科文件
        for root, _, files in os.walk(processed_data_path):
            for file in files:
                if not file.startswith('.') and not file.startswith('~$'): # 排除隱藏檔，如 Office 產生的暫存檔（~$ 開頭）
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, processed_data_path)
                    # 取得原始檔案大小 (Bytes) 並轉換為 MB
                    raw_size_mb = os.path.getsize(abs_path) / (1024 * 1024) 
                    # 乘以 100 進行無條件進位後，再除以 100，保留到小數點後兩位
                    size_mb = math.ceil(raw_size_mb * 100) / 100
                    mtime_str = datetime.fromtimestamp(os.path.getmtime(abs_path)).strftime('%Y-%m-%d %H:%M') # 取得檔案最後修改時間(檔案上傳時間)
                    
                    file_data.append({"選取刪除": False, "檔案路徑": rel_path, "檔案大小 (MB)": size_mb, "檔案上傳時間": mtime_str})

        if file_data:
            st.write(f"目前資料庫內共有 {len(file_data)} 筆可檢索檔案：")
            df = pd.DataFrame(file_data)
            # 使用 data_editor 產生帶有核取方塊的表格
            # disabled 是用於鎖定其他欄位，避免改到文字內容
            edited_df = st.data_editor(df, column_config={"選取刪除": st.column_config.CheckboxColumn("標記刪除", default=False)}, disabled=["檔案路徑", "檔案大小 (MB)", "檔案上傳時間"], hide_index=True, width="stretch")
            
            # 過濾出被勾選要刪除的檔案
            selected_files = edited_df[edited_df["選取刪除"] == True]["檔案路徑"].tolist()
            
            if selected_files:
                st.warning(f"⚠️ 您已選取 {len(selected_files)} 個檔案。")
                if st.button("確認刪除並更新資料庫", type="primary", width="stretch"):
                    with st.spinner("正在實體刪除檔案並清理向量資料庫..."):
                        for rel_path in selected_files:
                            file_path = os.path.join(processed_data_path, rel_path)
                            try:
                                # 1. 先從 Chroma 資料庫中移除該檔案的檢索塊
                                remove_file_from_db(file_path, st.session_state.vectordb)
                                # 2. 將硬碟中的實體檔案刪除
                                os.remove(file_path)
                            except Exception as e:
                                st.error(f"刪除失敗: {rel_path}, 錯誤: {e}")
                    
                    # 刪除完成後，寫入成功訊息並重新整理畫面
                    st.session_state["show_success_toast"] = f"✅ 已成功移除 {len(selected_files)} 個檔案！"
                    st.rerun()
                    
    # --- 分頁 3：檔案摘要預覽功能 ---
    with tab_preview:
        preview_data = []
        
        # 掃描資料夾內現有的文件，整理成列表顯示
        for root, _, files in os.walk(processed_data_path):
            for file in files:
                if not file.startswith('.') and not file.startswith('~$'):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, processed_data_path)
                    # 取得原始檔案大小 (Bytes) 並轉換為 MB
                    raw_size_mb = os.path.getsize(abs_path) / (1024 * 1024) 
                    # 乘以 100 進行無條件進位後，再除以 100，保留到小數點後兩位
                    size_mb = math.ceil(raw_size_mb * 100) / 100
                    mtime_str = datetime.fromtimestamp(os.path.getmtime(abs_path)).strftime('%Y-%m-%d %H:%M')
                    
                    preview_data.append({
                        "檔案路徑": rel_path, 
                        "檔案大小 (MB)": round(size_mb, 3), 
                        "檔案上傳時間": mtime_str
                    })

        if preview_data:
            st.markdown("💡 請勾選下方表格中的任意檔案，AI 助手將為您快速生成內容摘要：", help="使用模型：`gemma3:4b`")
            df_preview = pd.DataFrame(preview_data)
            
            preview_event = st.dataframe(
                df_preview, 
                on_select="rerun", 
                selection_mode="single-row", 
                hide_index=True, 
                width="stretch"
            )
            
            selected_preview_rows = preview_event.selection.rows
            
            if selected_preview_rows:
                selected_file_name = df_preview.iloc[selected_preview_rows[0]]["檔案路徑"]
                file_path = os.path.join(processed_data_path, selected_file_name)
                ext = os.path.splitext(file_path)[1].lower()
                
                st.divider()
                st.markdown(f"### 已選取檔案：`{selected_file_name}`")
                
                # 建立 Session State 快取鍵值，用檔案名稱區隔
                cache_key = f"summary_cache_{selected_file_name}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = None
                
                # 宣告一個開關，用來控制「現在是否要執行生成邏輯」
                needs_generation = False
                
                # ---------------------------------------------------------
                # 狀況一：該檔案還沒生成過摘要，顯示提示與觸發按鈕
                # ---------------------------------------------------------
                if st.session_state[cache_key] is None:
                    if st.button("開始生成 AI 摘要", type="primary", use_container_width=True):
                        needs_generation = True
                
                # ---------------------------------------------------------
                # 狀況二：快取中已有摘要紀錄，顯示「重新生成」按鈕與靜態快取內容
                # ---------------------------------------------------------
                else:
                    if st.button("🔄 重新生成摘要", use_container_width=True, key=f"reset_{selected_file_name}"):
                        needs_generation = True
                    else:
                        # 只有在沒有按下重新生成時，才顯示舊的快取內容
                        cached_data = st.session_state[cache_key]
                        
                        with st.container(border=True):
                            st.markdown(cached_data["summary"])
                        
                        with st.expander("點此查看檔案內容"):
                            if ext == '.csv':
                                df_csv = pd.read_csv(file_path, encoding='utf-8-sig')
                                st.dataframe(df_csv, use_container_width=True)
                            else:
                                st.text(cached_data["truncated_text"])

                # ---------------------------------------------------------
                # 共用的生成邏輯 (只要任一按鈕被按下，就會走到這裡開始生成)
                # ---------------------------------------------------------
                if needs_generation:
                    raw_text = ""
                    try:
                        with st.spinner("正在讀取文件並生成摘要中，請稍候..."):
                            # 1. 根據不同格式提取原始文字
                            if ext == '.pdf':
                                docs = PyPDFLoader(file_path).load()
                                raw_text = "\n".join([doc.page_content for doc in docs])
                            elif ext in ['.docx', '.doc']:
                                docs = Docx2txtLoader(file_path).load()
                                raw_text = "\n".join([doc.page_content for doc in docs])
                            elif ext == '.csv':
                                df_csv = pd.read_csv(file_path, encoding='utf-8-sig')
                                raw_text = df_csv.head(50).to_string() 
                            else:
                                st.info("目前不支援此格式的摘要生成。")

                            # 2. 執行文本壓縮與清理
                            text_content = ""
                            if raw_text:
                                clean_text = re.sub(r'\n\s*\n+', '\n\n', raw_text)
                                text_content = clean_text.strip()

                            # 3. 呼叫 LLM 進行摘要並動態呈現
                            if text_content:
                                truncated_text = text_content[:3000]
                                summary_llm = get_llm("gemma3:4b", keep_alive_time="30s")
                                
                                summary_prompt = f"""
                                你是一位專業的醫院病理科 AI 助手。請幫我閱讀以下文件內容，並用繁體中文產出一份精簡、易讀的「內容摘要」。
                                請遵循以下規範：
                                1. 使用有條理的條列式清單呈現文件的主要重點。
                                2. 語氣保持專業與客觀。
                                3. 不要加入文件以外的個人猜測。
                                
                                【文件內容】：
                                {truncated_text}
                                """
                                
                                with st.container(border=True):
                                    ans = st.write_stream(summary_llm.stream(summary_prompt))
                                
                                # 將生成的摘要直接更新進快取中！
                                st.session_state[cache_key] = {
                                    "summary": ans,
                                    "truncated_text": truncated_text
                                }
                                
                                # 建立檢視原始檔案文字折疊選單
                                with st.expander("點此查看檔案內容"):
                                    if ext == '.csv':
                                        df_csv = pd.read_csv(file_path, encoding='utf-8-sig')
                                        st.dataframe(df_csv, use_container_width=True)
                                    else:
                                        st.text(truncated_text)
                                        
                            else:
                                st.warning("無法提取有效文字，檔案可能是純圖片或內容為空。")
                    except Exception as e:
                        st.error(f"摘要生成過程中發生錯誤：{e}")
            else:
                st.info("ℹ️ 尚未選擇檔案。請點擊上方表格中的任一列來載入摘要。")
        else:
            st.info("目前資料庫內沒有任何檔案可以預覽。")

# ==========================================
# 5. 重建資料庫:二次確認對話框
# ==========================================
# 使用 st.dialog 建立彈出式視窗，避免使用者誤觸按鈕就直接重建
@st.dialog("⚠️ 警告：強制重建資料庫")
def confirm_rebuild_dialog():
    
    main_container = st.empty() # 建立一個空容器
    
    # 將警告文字放進這個容器中
    with main_container.container():
        st.error("您確定要清空並重建整個病理科知識庫嗎？")
        st.write("這項操作將會：")
        st.write("1. 刪除目前資料庫中的所有檢索索引。")
        st.write("2. 重新掃描硬碟中所有的檔案並重新建立索引。")
        st.write("此過程可能需要數分鐘的時間，且期間系統無法進行問答。")
        
        # 設定取消與確認按鈕
        col1, col2 = st.columns(2)
        with col1:
            cancel_btn = st.button("取消操作", use_container_width=True)
        with col2:
            confirm_btn = st.button("確認重建", type="primary", use_container_width=True)

    # 如果點擊了取消，直接重新執行網頁，對話框會自動關閉
    if cancel_btn:
        st.rerun() 

    # 如果點擊了確認重建
    if confirm_btn:
        # 清空主畫面容器
        main_container.empty()
        
        # 在清空後的畫面上，建立一個新的佔位符，用來顯示更新的進度文字
        ui_placeholder = st.empty()
        
        # 傳入佔位符，開始執行重建並顯示進度
        with st.spinner("系統正在重建向量資料庫，請稍候..."):
            st.session_state.vectordb = build_vector_db(st.session_state.vectordb, ui_placeholder)
            
        st.session_state["show_success_toast"] = "✅ 向量資料庫重建完成！"
        st.rerun()      


# ==========================================
# 6. 主要網頁介面 (側邊欄與問答區)
# ==========================================
st.title("🔬 病理科檢索問答系統")
st.divider()

# 1. --- 側邊欄設計 ---
# 初始化身分狀態，預設為一般使用者
if "role" not in st.session_state:
    st.session_state.role = "user"

with st.sidebar:
    st.header("系統狀態")

    # 根據身分狀態，決定顯示什麼畫面
    if st.session_state.role == "user":
        # === 一般使用者看到的畫面 ===
        st.info("目前身分：一般使用者")

        with st.expander("系統管理員登入"):
            # 使用 type="password" 隱藏輸入的文字
            admin_pwd = st.text_input("請輸入管理員解鎖碼", type="password")
            if st.button("解鎖管理員權限", use_container_width=True):
                # 讀取 secrets.toml 檔案(.streamlit/secrets.toml) 中設定的密碼
                if admin_pwd == st.secrets["admin_password"]:
                    st.session_state.role = "admin"
                    st.rerun() # 密碼正確，重新整理畫面以顯示被隱藏的按鈕
                else:
                    st.error("密碼錯誤，請重新輸入！")
                    
    elif st.session_state.role == "admin":
        # === 管理員解鎖後看到的畫面 ===
        st.success("目前身分：系統管理員")
        
        # 把原本的檔案管理跟重建資料庫按鈕，搬到管理員模式
        st.subheader("管理員功能")
        if st.button("開啟檔案管理", icon="📁", use_container_width=True, key="admin_file_btn"):
            file_management_center() 
            
        if st.button("強制重建向量資料庫", type="primary", use_container_width=True, key="admin_rebuild_btn"):
            confirm_rebuild_dialog()
        
        
        # 提供登出按鈕，方便用完後隨時上鎖
        if st.button("登出管理員", use_container_width=True, key="admin_logout_btn"):
            st.session_state.role = "user"
            st.rerun()

    
    
    # 接收並顯示來自其他操作（如上傳、刪除、重建）的全域成功提示訊息
    # 透過 st.session_state 傳遞訊息，顯示完畢後立刻刪除，避免重新整理網頁時又重複彈出
    if "show_success_toast" in st.session_state:
        st.toast(st.session_state["show_success_toast"])
        del st.session_state["show_success_toast"] 
    
    st.divider() 
    
    st.header("檢索設定")
    # 模型選擇(下拉式選單)
    selected_model = st.selectbox(
        "選擇 LLM 模型", 
        options=["gemma3:4b", "weitsung50110/llama-3-taiwan:8b-instruct-dpo-q8_0", "gpt-oss:20b"], 
        index=1 # 預設選擇第二個(Llama-3-Taiwan)
        )
    
    
    # 檢索相似度門檻
    score_threshold = st.slider(
        "檢索相似度門檻", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, # 預設為 0.3
        step=0.05,
        help="設定檢索參考資料的標準。數值越高，系統只會採納與問題高度相關的文字（若設太高可能導致找不到資料）；數值越低，則會放寬標準，抓取更多邊緣相關的內容。")
    
    # 檢索數量
    top_k_setting = st.slider(
        "檢索文本數量", 
        min_value=1, 
        max_value=6, 
        value=4, # 預設為 4，避免一次塞入過多文本導致 LLM 產生注意力遺失
        step=1)
    
    st.divider()
    
    st.header("記憶設定")
    enable_memory = st.toggle("啟用記憶模式", value=True, help="開啟後，系統會記住近期的對話內容。")
    
    
    # 建立一個有邊框的容器，把隸屬記憶模式的子設定包起來
    with st.container(border=True):
        # 記憶輪數設定 (連動記憶模式)
        memory_rounds = st.slider(
            "對話記憶輪數", 
            min_value=1, 
            max_value=5, 
            value=3, # 預設記憶近 3 輪對話
            step=1, 
            disabled=not enable_memory, # 當記憶模式關閉時，鎖定滑桿無法拖曳
            help="設定系統要參考過去幾輪的對話（一問一答為一輪）。"
        )
    
        # 2. 子選項：問題改寫模式 (連動記憶模式)
        enable_rewrite = st.toggle(
            "啟用問題改寫模式", 
            value=True, 
            # 必須開啟記憶模式，改寫功能才有歷史對話可以參考，否則自動鎖定
            disabled=not enable_memory, 
            help="開啟後，會利用 LLM 將代名詞替換為完整提問。若未開啟記憶，此功能會自動停用。"
        )
    
    # 清除記憶按鈕：清空 session_state 內的 messages 陣列，並重新刷新畫面
    if st.button("清除目前對話記憶/紀錄"):
        st.session_state.messages = []
        st.rerun() 
   
    

  
# 實體化目前選取的 LLM
# 讀取側邊欄的「模型名稱」
# 只要這個參數不變動，Streamlit 就不會重複浪費時間去載入模型
llm = get_llm(selected_model) 

# 初始化 Session State (用來保存網頁重新整理也不會消失的資料)
# Streamlit 的特性是每次使用者互動（如點擊按鈕、輸入文字），整個程式碼都會從頭執行一次
# 因此必須利用 st.session_state 建立一個「暫存記憶」，用來保存跨回合不會消失的資料
if "messages" not in st.session_state:
    st.session_state.messages = [] # 初始化空陣列，用來存放歷史對話紀錄
if "vectordb" not in st.session_state:
    st.session_state.vectordb = get_vector_db() # 建立向量資料庫連線，避免每次重整都重新讀取硬碟

# 歷史對話紀錄與參考文獻
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # 顯示對話文字內容
        
        # 檢查該筆歷史紀錄是否有檢索來源資料 (docs)
        if "docs" in message and message["docs"]:
            st.markdown("#### 🔍 檢索來源片段：")
            for i, doc in enumerate(message["docs"]):
                source_name = os.path.basename(doc.metadata.get('source', '未知'))
                page_num = doc.metadata.get('page')
                
                # 如果是 PDF 通常會有頁碼資訊，一併顯示出來
                if page_num is not None:
                    header_text = f"來源 {i+1}: {source_name} (第 {page_num + 1} 頁)"
                else:
                    header_text = f"來源 {i+1}: {source_name}"
                
                # 建立可摺疊的面板，讓版面保持乾淨，不會被長篇的醫療文本塞滿
                with st.expander(header_text):
                    st.markdown(f'<div style="font-size: 0.85em; color: #505050;">{doc.page_content}</div>', unsafe_allow_html=True)
        
        
        


# ==========================================
# 7. 檢索問答處理流程
# ==========================================
# 當使用者在對話框輸入問題時觸發
if prompt_input := st.chat_input("請輸入關於病理科流程的問題..."):
    # 1. 將使用者的問題存入歷史對話並顯示在畫面上
    st.session_state.messages.append({"role": "user", 
                                      "content": prompt_input,
                                      })
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        # 檢查是否有實際的檔案存在 (排除隱藏檔)，如果資料庫裡根本沒檔案，就中斷執行並提示使用者上傳
        valid_files = [f for f in os.listdir(processed_data_path) if not f.startswith('.') and not f.startswith('~')]
        if not valid_files:
            st.error("目前資料庫中沒有任何文件，請先點擊左側「開啟檔案管理」上傳病理科檔案。")
            st.stop() # 停止往下執行
        
        # 取得最近幾筆對話紀錄作為上下文參考
        chat_history_str = ""
        
        # 只有在「啟用記憶模式」且歷史訊息數量大於 1（代表有過去的對話）時才進行擷取
        if enable_memory and len(st.session_state.messages) > 1:
            # 根據使用者設定的輪數計算要抓取的訊息筆數 (一輪包含 user 和 assistant 兩筆，所以乘以 2)
            # -1 是為了排除當下使用者剛輸入的最新問題
            history_length = memory_rounds * 2
            
            # 使用切片語法 [-(history_length + 1):-1] 往回抓取對話
            # 結尾 -1 的用意是「排除掉使用者剛剛輸入的最新問題」，只保留純歷史紀錄
            history_messages = st.session_state.messages[-(history_length + 1):-1]
            
            for msg in history_messages:
                role_name = "病理科助手" if msg["role"] == "assistant" else "使用者"
                chat_history_str += f"{role_name}：{msg['content']}\n"
        
        # --- 查詢改寫 (Query Rewriting) ---
        # 解決代名詞問題：例如使用者上一句問「細胞學檢查」，下一句問「這個流程要多久？」
        # 向量檢索看不懂「這個」，所以要請 LLM 幫忙把問題改寫成「細胞學檢查流程要多久？」
        # 只有在「啟用問題改寫」且「存在歷史對話紀錄」時，才呼叫 LLM 進行改寫
        if enable_rewrite and chat_history_str.strip():
            rewrite_prompt = (
                f"你是一位精通醫院病理科專業術語的檢索詞優化專家"
                f"請根據使用者的【歷史對話紀錄】，判斷並將使用者的【最新問題】轉換為適合在向量資料庫中搜尋的「獨立關鍵句」。"
                f"""改寫原則：
                    1. 補齊代名詞：如果最新問題中包含「這個」、「該項檢查」、「他」等代名詞，請務必替換成歷史對話中對應的具體病理科名詞（例如：免疫組織化學染色、細胞學檢查等）。
                    2. 保持原意：若最新問題已經是一個完整且獨立的提問，不需要歷史對話補充即可理解，請直接輸出使用者的【最新問題】，絕對不要過度添加無關的詞彙。
                    3. 輸出格式：只能輸出一個最終的搜尋語句，嚴禁包含任何解釋、問候語或「改寫後的問題為：」等開場白。"""
                    f"【歷史對話紀錄】：{chat_history_str}"
                    f"【最新問題】：「{prompt_input}」"
            )
            
            # 呼叫 LLM 執行一次對話生成，並用 strip() 去除前後多餘空白
            search_query = llm.invoke(rewrite_prompt).strip()
            # 將改寫後的問題顯示在畫面下方（以小灰字呈現）
            st.caption(f"🔍 Query Rewriting：{search_query}") 
        else:
            search_query = prompt_input # 沒開記憶或沒有對話紀錄時，直接使用原問題搜尋
        
        # --- 向量資料庫檢索 ---
        # 將 Chroma 資料庫轉換為 LangChain 支援的檢索器格式
        retriever = st.session_state.vectordb.as_retriever(
            search_type="similarity_score_threshold", # 採用相似度門檻過濾
            search_kwargs={'score_threshold': score_threshold, 'k': top_k_setting} # 套用側邊欄拉桿的參數
        )
        
        # 透過改寫過的問題 (search_query) 進行相似度搜尋
        # invoke 會回傳一個陣列，裡面包含數個文本(也就是我們前面切好的 600 字文本塊)
        docs = retriever.invoke(search_query) 

        # ==========================================
        # 🛑 防呆機制：檢索結果為空時直接阻斷 LLM 生成
        # ==========================================
        if not docs:
            # 1. 設定找不到資料時的預設回覆
            fallback_msg = "抱歉，目前病理科資料庫中，沒有檢索到與您問題相關的參考資料。請嘗試更換關鍵字或提供更具體的流程名稱。"
            
            # 2. 直接將預設訊息輸出到畫面上
            st.write(fallback_msg)
            
            # 3. 顯示來源片段區塊（填入查無資料）
            st.markdown("#### 🔍 檢索來源片段：")
            st.write("沒有找到相關的參考資料。")
            
            # 4. 為了讓「對話記憶模式」保持連貫，我們需要把這筆失敗的查詢也手動存入歷史紀錄中
            st.session_state.messages.append({
                "role": "assistant", 
                "content": fallback_msg,
                "docs": []  
            })
            
            # 5. 強制中斷 Streamlit 往下執行，這樣就不會去呼叫 LLM 浪費 Token 與時間
            st.stop()
        # ==========================================

        # 如果檢索到了資料，才繼續往下執行 LLM 生成回答的流程
        # 將找到的多段參考資料合併成一個大字串，準備餵給 LLM
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # --- 最終回答生成 ---
        # 組合 System Prompt，嚴格限制 LLM 的回答範圍，避免產生資訊幻覺
        system_prompt = (
            "你是一位專業的醫院病理科助手。你的任務是根據所提供的參考資料與歷史對話紀錄來回答問題。\n\n"
            "回答規範：\n"
            "1. 僅根據提供的參考資料進行回答。如果資料中沒有提到相關資訊，請誠實回答「抱歉，在目前的病理科規範資料中找不到相關資訊」，不可隨意編造。\n"
            "2. 若資料提及多項流程或步驟，請使用條列式清單呈現。\n"
            "3. 嚴禁根據你自身的訓練知識來補充資料庫以外的醫療建議或行政流程。\n"
            "4. 所有的回答必須使用繁體中文，並維持專業、嚴謹且親切的語氣。\n\n"
            "【歷史對話紀錄】\n"
            "{chat_history}\n\n"
            "【參考資料】\n"
            "{context}"
        )
        
        # 組合 System Prompt 與使用者的最終提問
        qa_prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            ('user', '問題: {input}')
        ])

        final_prompt = qa_prompt.format(
            chat_history=chat_history_str,
            context=context_text, 
            input=prompt_input 
        )
        
        # 產生串流回答
        # 用意：不需要等 LLM 把幾百個字全部想完才一次顯示
        def stream_generator():
            for chunk in llm.stream(final_prompt):
                yield chunk         
                
        try:
            ans = st.write_stream(stream_generator())
            
            st.markdown("#### 🔍 檢索來源片段：")
            
            
            for i, doc in enumerate(docs):
                source_name = os.path.basename(doc.metadata.get('source', '未知'))
                page_num = doc.metadata.get('page')
                    
                if page_num is not None:
                    header_text = f"來源 {i+1}: {source_name} (第 {page_num + 1} 頁)"
                else:
                    header_text = f"來源 {i+1}: {source_name}"
                        
                with st.expander(header_text):
                    # 統一使用縮小的深灰色字體
                    st.markdown(f'<div style="font-size: 0.85em; color: #505050;">{doc.page_content}</div>', unsafe_allow_html=True)
            
            
            # 處理完成後，將這次 AI 的回答以及對應的檢索文獻 (docs)，存進 Session State 暫存記憶中
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ans,
                "docs": docs  
            })
            
        except Exception as e:
            # 錯誤處理：如果 Ollama 沒開、本機記憶體不足、或是模型名稱輸入錯誤等異常狀態
            st.error("⚠️ 無法連線至 LLM 模型。請確認本機端的 Ollama 服務已經啟動，或是模型名稱設定正確。")
            

            
st.markdown(
    """
    <style>
    .disclaimer-text {
        position: fixed;
        bottom: 5px; /* 距離網頁最底部的距離 */
        left: 0;
        right: 0;
        text-align: center;
        font-size: 12px;
        color: #888888;
        background-color: transparent;
        z-index: 999; /* 確保不會被其他元件遮擋 */
    }
    
    </style>
    <div class="disclaimer-text">
        ⚕️ 病理科助手為 AI 輔助系統，有時可能產生錯誤資訊。結果僅供參考，請務必與專業醫療規範確認。
    </div>
    """,
    unsafe_allow_html=True
)           
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            