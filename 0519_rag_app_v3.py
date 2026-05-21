import os
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
def get_llm(model_name, temperature_value):
    return OllamaLLM(
        model=model_name,
        num_ctx=8192,  # 如未設定，Ollama預設是 2048 Tokens
        temperature=temperature_value 
    )

# 載入嵌入模型 (Embeddings)
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model='bge-m3')
    
embeddings = get_embeddings()

# ==========================================
# 2. 定義路徑
# ==========================================
# 取得目前檔案所在的資料夾絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 存放使用者上傳並轉檔完成的檔案路徑
processed_data_path = os.path.join(BASE_DIR, "01_processed_data") 
# 存放 Chroma 向量資料庫的路徑  
chroma_path = os.path.join(BASE_DIR, "02_db", "chroma_db")       
# 如果存放處理資料的資料夾不存在，則自動建立一個
os.makedirs(processed_data_path, exist_ok=True)

# ==========================================
# 3. 向量資料庫操作 (支援局部新增/刪除)
# ==========================================
def get_vector_db():
    """讀取現有的資料庫"""
    return Chroma(persist_directory=chroma_path, embedding_function=embeddings)

def add_file_to_db(file_path, vectordb):
    """將單一檔案切塊並新增至 Chroma 資料庫，
    不用每次上傳新檔案就全部重新建立資料庫"""
    ext = os.path.splitext(file_path)[1].lower() # 取得副檔名並轉小寫
    
    # 根據副檔名選擇對應的載入器 (Loader)
    if ext == '.pdf':
        docs = PyPDFLoader(file_path).load()
    elif ext == '.docx':
        docs = Docx2txtLoader(file_path).load()
    elif ext == '.csv':
        docs = CSVLoader(file_path, encoding='utf-8-sig').load() # 使用 utf-8-sig 避免中文亂碼
    else:
        return False 
    
    if docs:
        # 進行文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,     
            chunk_overlap=150,  # 重疊 150 字
            separators=["\n\n", "\n", "。", "，", " ", ""]  
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
    """強制重整： 掃描 01_processed_data 並重建 Chroma 資料庫"""
    
    log_text = "### 🔄 向量資料庫重建進度\n\n"
    
    def update_ui(msg):
        """內部小工具：僅更新網頁畫面，不顯示於終端機"""
        nonlocal log_text
        log_text += f"{msg}\n\n"
        if ui_placeholder:
            ui_placeholder.markdown(log_text)
    
    
    try:
        # 清空向量資料庫內的所有資料，而不是刪除實體資料夾，避免遇到「檔案被鎖定」的報錯
        update_ui("✅ **[階段 1/4]** 正在清空舊有向量資料庫...")
        vectordb.delete_collection()
    except Exception as e:
        pass # 如果是空資料庫或初次建立，可能沒有資料可以清，直接略過

    # 重新讀取資料夾內所有支援的檔案
    update_ui("✅ **[階段 2/4]** 正在掃描並讀取硬碟中的病理科檔案...")
    pdf_docs = DirectoryLoader(processed_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    docx_docs = DirectoryLoader(processed_data_path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
    csv_docs = DirectoryLoader(processed_data_path, glob="**/*.csv", loader_cls=CSVLoader, loader_kwargs={'encoding': 'utf-8-sig'}).load()

    docs = pdf_docs + docx_docs + csv_docs

    # 若沒有任何文件，回傳空的 Chroma 實例
    if not docs:
        return get_vector_db()

    update_ui("✅ **[階段 3/4]** 正在進行文本分割...")    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  
        chunk_overlap=150,  
        separators=["\n\n", "\n", "。", "，", " ", ""]  
    )
    split_docs = text_splitter.split_documents(docs)

    # 建立並儲存新的 Chroma 向量資料庫
    update_ui("✅ **[階段 4/4]** 嵌入模型轉換向量並寫入資料庫...")
    new_vectordb = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings,
        persist_directory=chroma_path 
    )
    return new_vectordb   

# ==========================================
# 4. 檔案管理介面 (使用 Streamlit 對話框功能)
# ==========================================
@st.dialog("📁 檔案管理", width="large")
def file_management_center():
    """檔案管理中心：包含上傳新文件與管理現有文件(刪除)的頁籤"""
    tab_upload, tab_manage = st.tabs(["📤 上傳新文件", "🗑️ 管理現有文件"])

    # --- 分頁 1：上傳功能 ---
    with tab_upload:
        st.write("支援檔案類型: pdf, docx, doc, csv, xls, xlsx")
        uploaded_files = st.file_uploader(
            "選擇檔案", 
            type=['pdf', 'docx', 'doc', 'csv', 'xls', 'xlsx'], # 限定上傳檔案類型
            accept_multiple_files=True, # 支援多檔案上傳
            key="dialog_uploader"
            )
        
        if st.button("上傳並更新資料庫", icon="💾", use_container_width=True):
            if uploaded_files:
                save_count = 0
                
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
                        
                        # 即時更新網頁上的狀態文字
                        current_step = idx + 1
                        status_text.write(f"⏳ 正在處理 ({current_step}/{total_files})：**{file.name}** ...")
                        
                        file_path = os.path.join(processed_data_path, file.name)
                        name, ext = os.path.splitext(file.name)
                        ext = ext.lower()
                        
                        # 1. --- 攔截加密檔案 ---
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
                            # 加入失敗清單
                            failed_records.append({"檔案名稱": file.name, "失敗原因": "檔案已加密，請解除密碼後再上傳"})
                            continue # 發現加密，直接跳過這個檔案，不寫入硬碟也不轉檔
              
                
                        # 2. --- 檢查重複檔名 (安全阻擋機制) ---
                        # 掃描資料夾內是否有相同主檔名的檔案 (例如找 細胞學檢查規範.*)
                        search_pattern = os.path.join(processed_data_path, f"{name}.*")
                        existing_files = glob.glob(search_pattern)
                        
                        # 如果上傳的是 Excel，也要檢查是否已經有拆解出來的 CSV 檔
                        search_pattern_csv = os.path.join(processed_data_path, f"{name}_*.csv")
                        existing_files.extend(glob.glob(search_pattern_csv))
                        
                        # 如果找到任何同名的舊檔案，立刻攔截！
                        if existing_files:
                            failed_records.append({
                                "檔案名稱": file.name, 
                                "失敗原因": "已存在同名檔案。為避免資料遺失，請先至「管理現有文件」手動刪除舊檔後再上傳"
                            })
                            continue # 直接跳過這個檔案，不寫入硬碟也不轉檔，繼續處理下一個檔案
                            
                        # 3. --- 正式寫入硬碟 ---
                        # 確定沒有同名衝突後，才將新上傳的檔案寫入硬碟
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        
                        # 4. --- 轉檔與資料庫寫入 ---
                        try:
                            # 處理舊版 Word (.doc) -> 呼叫 LibreOffice 轉成 .docx
                            if ext == '.doc':
                                soffice_path = shutil.which('libreoffice') or shutil.which('soffice') or r"C:\Program Files\LibreOffice\program\soffice.exe"
                                subprocess.run([soffice_path, '--headless', '--convert-to', 'docx', '--outdir', processed_data_path, file_path], check=True, timeout=60)
                                os.remove(file_path) # 轉檔成功後刪除原始 .doc
                                
                                new_docx_path = os.path.join(processed_data_path, f"{name}.docx")
                                if add_file_to_db(new_docx_path, st.session_state.vectordb):
                                    save_count += 1
                                else:
                                    os.remove(new_docx_path) 
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "轉檔後無法提取純文字"})
                                
                            # 處理 Excel (.xls, .xlsx) -> 拆解成多個 CSV 工作表
                            elif ext in ['.xls', '.xlsx']:
                                excel_dict = pd.read_excel(file_path, sheet_name=None, dtype=str)
                                valid_csv_count = 0
                                for sheet_name, df in excel_dict.items():
                                    # 移除全空的欄與列
                                    df.dropna(how='all', inplace=True)
                                    df.dropna(how='all', axis=1, inplace=True)
                                    if not df.empty:
                                        csv_filename = f"{name}_{sheet_name}.csv"
                                        csv_path = os.path.join(processed_data_path, csv_filename)
                                        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                                        
                                        if add_file_to_db(csv_path, st.session_state.vectordb):
                                            valid_csv_count += 1
                                        else:
                                            os.remove(csv_path) 
                                            
                                os.remove(file_path) # 拆解完成後刪除原始 Excel
                                if valid_csv_count > 0:
                                    save_count += 1
                                else:
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "無法提取有效表格資料"})
                            
                            # 不需要轉檔的 PDF, DOCX, CSV 直接寫入向量資料庫
                            else:
                                if add_file_to_db(file_path, st.session_state.vectordb):
                                    save_count += 1
                                else:
                                    os.remove(file_path) 
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "純圖片或無法解析的內容"})
                                
                        except Exception as e:
                            # 如果過程中發生錯誤，清理剛剛寫入的殘留檔案
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
                    # 情況二：有失敗情況，顯示資料表讓使用者知道哪些檔案有問題
                    if save_count > 0:
                        st.success(f"✅ 已成功寫入 {save_count} 份檔案。")
                    
                    st.error(f"⚠️ 發現 {len(failed_records)} 份檔案無法寫入，上傳失敗，請人工確認內容：")
                    
                    # 利用 Pandas DataFrame 呈現乾淨的表格，hide_index=True 去除最左邊的數字序號
                    df_failed = pd.DataFrame(failed_records)
                    st.dataframe(df_failed, hide_index=True, use_container_width=True)
            else:
                st.error("請先選擇要上傳的檔案。")

    # --- 分頁 2：管理/刪除功能 ---
    with tab_manage:
        file_data = []
        
        # 掃描資料夾內現有的文件，整理成列表顯示
        for root, _, files in os.walk(processed_data_path):
            for file in files:
                if not file.startswith('.') and not file.startswith('~$'):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, processed_data_path)
                    size_mb = os.path.getsize(abs_path) / (1024 * 1024) # 顯示檔案大小(MB)
                    mtime_str = datetime.fromtimestamp(os.path.getmtime(abs_path)).strftime('%Y-%m-%d %H:%M') # 顯示檔案修改時間
                    
                    file_data.append({"選取刪除": False, "檔案路徑": rel_path, "檔案大小 (MB)": round(size_mb, 3), "檔案上傳時間": mtime_str})

        if file_data:
            st.write(f"目前資料庫內共有 {len(file_data)} 筆可檢索檔案：")
            df = pd.DataFrame(file_data)
            # 使用 data_editor 產生帶有核取方塊的表格
            edited_df = st.data_editor(df, column_config={"選取刪除": st.column_config.CheckboxColumn("標記刪除", default=False)}, disabled=["檔案路徑", "檔案大小 (MB)", "修改時間"], hide_index=True, width="stretch")
            
            # 過濾出被勾選要刪除的檔案
            selected_files = edited_df[edited_df["選取刪除"] == True]["檔案路徑"].tolist()
            
            if selected_files:
                st.warning(f"⚠️ 您已選取 {len(selected_files)} 個檔案。")
                if st.button("確認刪除並更新資料庫", type="primary", width="stretch"):
                    with st.spinner("正在實體刪除檔案並清理向量資料庫..."):
                        for rel_path in selected_files:
                            file_path = os.path.join(processed_data_path, rel_path)
                            try:
                                # 1. 從 Chroma 資料庫中移除該檔案的檢索塊
                                remove_file_from_db(file_path, st.session_state.vectordb)
                                # 2. 將硬碟中的實體檔案刪除
                                os.remove(file_path)
                            except Exception as e:
                                st.error(f"刪除失敗: {rel_path}, 錯誤: {e}")
                                
                    st.session_state["show_success_toast"] = f"✅ 已成功移除 {len(selected_files)} 個檔案！"
                    st.rerun()

# ==========================================
# 5. 重建資料庫:二次確認對話框
# ==========================================
@st.dialog("⚠️ 警告：強制重建資料庫")
def confirm_rebuild_dialog():
    
    main_container = st.empty()
    
    # 將原本的內容放進這個容器中
    with main_container.container():
        st.error("您確定要清空並重建整個病理科知識庫嗎？")
        st.write("這項操作將會：")
        st.write("1. 刪除目前資料庫中的所有檢索索引。")
        st.write("2. 重新掃描硬碟中所有的檔案並重新建立索引。")
        st.write("此過程可能需要數分鐘的時間，且期間系統無法進行問答。")
        
        col1, col2 = st.columns(2)
        with col1:
            # 將按鈕存成變數，把邏輯判斷拉到外面
            cancel_btn = st.button("取消操作", use_container_width=True)
        with col2:
            confirm_btn = st.button("確認重建", type="primary", use_container_width=True)

    # 如果點擊了取消
    if cancel_btn:
        st.rerun() 

    # 如果點擊了確認重建
    if confirm_btn:
        # 清空主畫面容器
        main_container.empty()
        
        # 在原本乾淨的畫面上，建立一個新的全寬度進度佔位符
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
with st.sidebar:
    st.subheader("📁 資料庫管理")
    if st.button("開啟檔案管理", icon="📁", use_container_width=True):
        file_management_center() # 呼叫上方的對話框
    
    # 接收並顯示來自其他操作的成功提示訊息
    if "show_success_toast" in st.session_state:
        st.toast(st.session_state["show_success_toast"])
        del st.session_state["show_success_toast"] 
    
    st.divider() 
    
    st.header("檢索設定")
    # 模型選擇
    selected_model = st.selectbox(
        "選擇 LLM 模型", 
        options=["gemma3:4b", "weitsung50110/llama-3-taiwan:8b-instruct-dpo-q8_0"], 
        index=1 # 預設選擇第二個
        )
    
    # 溫度(Temperature)選擇
    temperature_setting = st.slider(
        "溫度(Temperature)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0, # 預設為 0
        step=0.1,
        help="控制回答的發散程度。數值越低（接近 0），回答越精確、保守且穩定，最適合嚴謹的醫療問答；數值越高，會試圖發揮創意，可能增加產生錯誤資訊（幻覺）的風險。"
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
        value=4, # 預設為 4
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
            disabled=not enable_memory, # 當記憶模式關閉時，滑桿無法拖曳
            help="設定系統要參考過去幾輪的對話（一問一答為一輪）。"
        )
    
        # 2. 子選項：問題改寫模式 (連動記憶模式)
        enable_rewrite = st.toggle(
            "啟用問題改寫模式", 
            value=True, 
            disabled=not enable_memory, # 關鍵連動：當記憶關閉時，改寫功能會被反灰無法點擊
            help="開啟後，會利用 LLM 將代名詞替換為完整提問。若未開啟記憶，此功能會自動停用。"
        )
    
    if st.button("清除目前對話記憶/紀錄"):
        st.session_state.messages = []
        st.rerun() 
    
    st.divider()
    
    if st.button("強制重建向量資料庫", type="primary"):
        confirm_rebuild_dialog()
        
  
# 實體化目前選取的 LLM
llm = get_llm(selected_model, temperature_setting) 

# 初始化 Session State (用來保存網頁重新整理也不會消失的資料)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = get_vector_db()

# 歷史對話紀錄
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# 7. 檢索問答處理流程
# ==========================================
# 當使用者在對話框輸入問題時觸發
if prompt_input := st.chat_input("請輸入關於病理科流程的問題..."):
    # 1. 將使用者的問題存入歷史對話並顯示在畫面上
    st.session_state.messages.append({"role": "user", "content": prompt_input})
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
            
            # 使用切片取得指定長度的歷史訊息
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
            
            # 呼叫 LLM 執行改寫
            search_query = llm.invoke(rewrite_prompt).strip()
            # 將改寫後的問題顯示在畫面下方（以小灰字呈現）
            st.caption(f"🔍 Query Rewriting：{search_query}") 
        else:
            search_query = prompt_input # 如果沒有歷史對話，就直接使用原本的問題
        
        # --- 向量資料庫檢索 ---
        retriever = st.session_state.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': score_threshold, 'k': top_k_setting} 
        )
        
        # 透過改寫過的問題 (search_query) 進行相似度搜尋
        docs = retriever.invoke(search_query) 
        # 將找到的多段參考資料合併成一個大字串，準備餵給 LLM
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # --- 最終回答生成 ---
        # 組合 System Prompt，嚴格限制 LLM 的回答範圍，避免產生資訊幻覺
        system_prompt = (
            "你是一位專業的醫院病理科助手。你的任務是根據所提供的參考資料（Context）來回答問題。\n\n"
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
        def stream_generator():
            for chunk in llm.stream(final_prompt):
                yield chunk         
                
        try:
            ans = st.write_stream(stream_generator())
            
            # 使用折疊選單 (Expander) 顯示資料庫找出來的參考來源
            with st.expander("查看檢索來源片段"):
                if docs:
                    for i, doc in enumerate(docs):
                        st.write(f"**來源 {i+1}:** {os.path.basename(doc.metadata.get('source', '未知'))}")
                        st.caption(doc.page_content) 
                else:
                    st.write("沒有找到相關的參考資料。")
            
            # 將 LLM 的最終回答存入歷史紀錄，讓下一輪對話可以參考
            st.session_state.messages.append({"role": "assistant", "content": ans})
            
        except Exception as e:
            # 錯誤處理：如果 Ollama 沒開或是模型名字打錯
            st.error("⚠️ 無法連線至 LLM 模型。請確認本機端的 Ollama 服務已經啟動，或是模型名稱設定正確。")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
