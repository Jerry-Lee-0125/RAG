# 🔬 病理科檢索問答系統

這是一個基於檢索增強生成（RAG）技術開發的病理科專屬問答系統。本專案結合了大型語言模型（LLM）與向量資料庫，能夠自動解析多種格式的醫療文書與表單，並提供精確、具有引用來源的專業回答。

## 🌟 系統特色

* **本地化部署**：運用 Ollama 運行本地端 LLM，確保醫療資料的隱私與安全性。
* **多格式文件支援**：支援 `.pdf`, `.docx`, `.doc`, `.csv`, `.xls`, `.xlsx` 等常見病理科文書格式。
* **動態向量資料庫**：內建文件管理介面，支援單筆檔案的上傳、更新與刪除，無須每次重新建置整個資料庫。
* **智慧問題改寫**：具備對話記憶功能，能自動將使用者的代名詞替換為完整提問，提升檢索精準度。

---

## 💻 系統環境需求

本教學以 **Windows 11** 作業系統為例，並建議使用 **Anaconda** 進行環境管理。
* **建議 Python 版本**：3.11
* **使用工具**：Anaconda

---

## 🛠️ 環境建置步驟 (Windows 11)

### 步驟一：安裝並設定 Ollama 與本地模型

本系統高度依賴本地端的大型語言模型與嵌入模型（Embedding Model）。請先確保您的電腦已安裝 Ollama。

1.  前往 [Ollama 官方網站](https://ollama.com/) 下載並安裝 Windows 版本。
2.  安裝完成後，開啟「命令提示字元 (CMD)」或「PowerShell」。
3.  拉取（Pull）系統所需的嵌入模型（BGE-M3）：
    ```bash
    ollama pull bge-m3
    ```
4.  拉取專案預設使用的 LLM 模型（可擇一或皆下載）：
    ```bash
    ollama pull gemma3:4b
    ollama pull weitsung50110/llama-3-taiwan:8b-instruct-dpo-q8_0
    ```
    *(註：確保 Ollama 服務在背景持續運行，系統才能順利呼叫模型)*

### 步驟二：建立 Anaconda 虛擬環境

為避免套件版本衝突，建議為本專案建立獨立的虛擬環境。

1.  開啟 **Anaconda Prompt**。
2.  建立名為 `pathology_rag` 的虛擬環境，並指定 Python 3.11：
    ```bash
    conda create -n pathology_rag python=3.11 -y
    ```
3.  啟動虛擬環境：
    ```bash
    conda activate pathology_rag
    ```

### 步驟三：安裝 Python 依賴套件

本專案使用 LangChain 0.3.x 生態系、Chroma 向量資料庫與 Streamlit 網頁框架。

1.  將終端機路徑切換至本專案資料夾底下。
2.  透過 `requirements_chroma.txt` 一鍵安裝所有依賴套件：
    ```bash
    pip install -r requirements_chroma.txt
    ```

### 步驟四：安裝外部依賴軟體 (LibreOffice)

由於系統內建將舊版 Word 檔案（`.doc`）自動轉換為 `.docx` 的功能，底層需要呼叫 LibreOffice 進行無頭（Headless）轉檔。

1.  前往 [LibreOffice 官方網站](https://zh-tw.libreoffice.org/download/libreoffice-still/) 下載並安裝最新版軟體。
2.  請確保軟體安裝於系統預設路徑：`C:\Program Files\LibreOffice\program\soffice.exe`。若安裝於其他路徑，需手動修改主程式 `0519_rag_app_v3.py` 中的 `soffice_path` 變數。

---

## 🚀 啟動系統

所有環境與依賴配置完成後，即可啟動檢索問答系統：

1.  確認 Anaconda Prompt 已進入 `pathology_rag` 虛擬環境。
2.  確認 Ollama 應用程式已在系統工具列背景執行。
3.  輸入以下指令啟動 Streamlit 網頁服務：
    ```bash
    streamlit run 0519_rag_app_v3.py
    ```
4.  瀏覽器將自動開啟 `http://localhost:8501`。
5.  初次使用時，請點擊左側側邊欄的「📁 開啟檔案管理」，上傳您的病理科相關文件（如流程規範、儀器說明書等），系統便會自動將其切塊並寫入 Chroma 向量資料庫中。

---

## 📂 專案架構說明

* `0519_rag_app_v3.py`：系統主程式，包含 Streamlit UI、LangChain 處理流程與 Chroma 資料庫操作。
* `requirements_chroma.txt`：記錄所有 Python 依賴套件的清單。
* `01_processed_data/`：（自動生成）存放使用者上傳並轉檔完成的實體檔案。
* `02_db/chroma_db/`：（自動生成）存放 Chroma 建立的本地向量資料庫檔案。
