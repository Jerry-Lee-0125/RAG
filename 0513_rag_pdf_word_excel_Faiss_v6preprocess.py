import os
import shutil
import subprocess
import pandas as pd

def sync_and_convert_files(raw_dir, processed_dir, error_dir):
    """
    同步 01_raw_data 與 02_processed_data 檔案，同時比對修改時間與清理已刪除的檔案。
    如果有任何新增、更新或刪除，將回傳 True，更新向量資料庫。
    """
    print("\n--- 開始執行檔案前處理與更新檢查 ---")
    allowed_extensions = ['.pdf', '.docx', '.csv']
    has_changes = False
    
    # 新增紀錄表，用來記錄所有應該存在的標準化檔案路徑
    expected_processed_files = set()
    
    # ==========================================
    # 第一階段：正向掃描 (檢查 01_raw_data 裡的資料是否有被 "新增" 或 "更新")
    # ==========================================
    for root, dirs, files in os.walk(raw_dir):
        for filename in files:
            
            # 略過暫存檔與系統隱藏檔
            if filename.startswith('~$'):
                continue 
            
            raw_path = os.path.join(root, filename)
            rel_path = os.path.relpath(root, raw_dir)
            target_dir = os.path.join(processed_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True) 
            
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            display_name = os.path.join(rel_path, filename) if rel_path != "." else filename
            raw_mtime = os.path.getmtime(raw_path)
            
            if ext in allowed_extensions:
                processed_path = os.path.join(target_dir, filename)
                # 將路徑記錄下來 (使用絕對路徑避免比對錯誤)
                expected_processed_files.add(os.path.abspath(processed_path))
                
                if not os.path.exists(processed_path) or raw_mtime > os.path.getmtime(processed_path):
                    shutil.copy2(raw_path, processed_path)
                    print(f"🔄 已更新檔案: {display_name}")
                    has_changes = True
            
            
            # ==========================================
            # 處理 .doc 轉檔 .docx 
            # ==========================================
            elif ext == '.doc':
                processed_docx_path = os.path.join(target_dir, f"{name}.docx")
                # 將轉檔後的路徑記錄下來 (使用絕對路徑避免比對錯誤)
                expected_processed_files.add(os.path.abspath(processed_docx_path))
                
                if not os.path.exists(processed_docx_path) or raw_mtime > os.path.getmtime(processed_docx_path):
                    print(f"❗ 發現 .doc 檔案更新，正在重新轉檔: {display_name} ...")
                    try:
                        
                        # 自動尋找 LibreOffice 執行檔
                        soffice_path = shutil.which('soffice')
                        # 如果系統找不到，使用預設的絕對路徑作為備案
                        if not soffice_path:
                            soffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe" #請確認路徑是否正確
                        
                        command = [
                            soffice_path, 
                            '--headless', 
                            '--convert-to', 'docx', 
                            '--outdir', target_dir, 
                            raw_path
                        ]
                        subprocess.run(command, check=True, timeout=60, capture_output=True)
                        print(f"✅ 轉檔成功: {os.path.join(rel_path, name)}.docx" if rel_path != "." else f"✅ 轉檔成功: {name}.docx")
                        has_changes = True
                    except subprocess.TimeoutExpired:
                        print(f"❌ 轉檔超時跳過: {display_name}")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ 轉檔失敗 (請確認檔案未開啟/未加密，再上傳一次): {display_name}")
    
                        # 將有問題的檔案移至 error_data 隔離，避免下次重複執行
                        os.makedirs(error_dir, exist_ok=True)
                        quarantine_path = os.path.join(error_dir, filename)
                        shutil.move(raw_path, quarantine_path) 
                        print(f"已將問題檔案隔離至: {quarantine_path}，避免下次重複執行。")
                    except Exception as e:
                        print(f"❌ 轉檔錯誤 {display_name}: {e}")
            
            # ==========================================
            # 處理 .xls 與 .xlsx 轉檔 .csv 
            # ==========================================
            elif ext in ['.xls', '.xlsx']:
                try:
                    # 建立一個標記檔案，專門用來記錄該 Excel 的最後處理時間
                    marker_path = os.path.join(target_dir, f".{filename}.marker")
                    # 將標記檔加入預期清單，避免被「反向清理」機制誤刪
                    expected_processed_files.add(os.path.abspath(marker_path))
                    
                    # 判斷是否需要更新：標記檔不存在，或者 Excel 原始檔的修改時間比標記檔還新
                    if not os.path.exists(marker_path) or raw_mtime > os.path.getmtime(marker_path):
                        print(f"❗ 發現 Excel 檔案變動，正在重新轉檔: {display_name} ...")
                        
                        # 【預防性清理】先刪除目標資料夾中舊有的對應 CSV
                        # 避免有人在原始 Excel 刪除了某個工作表，但舊的 CSV 還殘留在資料夾中
                        for existing_file in os.listdir(target_dir):
                            if existing_file.startswith(f"{name}_") and existing_file.endswith(".csv"):
                                os.remove(os.path.join(target_dir, existing_file))
                                
                        # 讀取 Excel 所有工作表
                        excel_dict = pd.read_excel(raw_path, sheet_name=None)
                        valid_sheets_count = 0
                        
                        for sheet_name, df in excel_dict.items():
                            # 【資料清洗】刪除全空列與全空欄
                            df.dropna(how='all', inplace=True)
                            df.dropna(how='all', axis=1, inplace=True)
                            
                            # 確認清洗後還有資料才存檔
                            if not df.empty:
                                csv_filename = f"{name}_{sheet_name}.csv"
                                processed_csv_path = os.path.join(target_dir, csv_filename)
                                df.to_csv(processed_csv_path, index=False, encoding='utf-8-sig')
                                # 將確實有產生的 CSV 加入預期清單
                                expected_processed_files.add(os.path.abspath(processed_csv_path))
                                valid_sheets_count += 1
                                
                        # 轉檔完成後，更新（或建立）標記檔案的時間戳記
                        with open(marker_path, 'w') as f:
                            f.write("sync_completed")
                            
                        
                        print(f"✅ 轉檔成功: {filename} (共產出 {valid_sheets_count} 個有效 CSV)")
                        has_changes = True
                        
                    else:
                        # 如果檔案未變更：我們必須把「目前已存在的對應 CSV」加回紀錄表
                        # 否則它們會在第二階段的「反向清理」被當作多餘檔案刪除掉
                        for existing_file in os.listdir(target_dir):
                            if existing_file.startswith(f"{name}_") and existing_file.endswith(".csv"):
                                expected_processed_files.add(os.path.abspath(os.path.join(target_dir, existing_file)))
                                
                except Exception as e:
                    print(f"❌ Excel 處理失敗 {display_name}: {e}")
                    os.makedirs(error_dir, exist_ok=True)
                    quarantine_path = os.path.join(error_dir, filename)
                    shutil.move(raw_path, quarantine_path) 
                    print(f"已將問題檔案隔離至: {quarantine_path}")
            
            
            
            
            



    # ==========================================
    # 第二階段：反向清理 (檢查 01_raw_data 裡的資料是否有被 "刪除")
    # ==========================================
    for root, dirs, files in os.walk(processed_dir):
        for filename in files:
            processed_file_path = os.path.join(root, filename)
            abs_processed_path = os.path.abspath(processed_file_path)
            
            # 如果發現標準化資料區裡面的檔案「不在記錄上」
            if abs_processed_path not in expected_processed_files:
                os.remove(processed_file_path) # 實體刪除檔案
                
                rel_path = os.path.relpath(processed_file_path, processed_dir)
                print(f"🗑️ 發現原始檔已刪除，同步移除: {rel_path}")
                has_changes = True # 標記有變動，觸發後續 FAISS 向量資料庫重建
    
    # 清理空資料夾
    for root, dirs, files in os.walk(processed_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # 檢查該資料夾是否完全沒有檔案或子資料夾
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                
    print("--- 檔案檢查完畢 ---\n")
    return has_changes
