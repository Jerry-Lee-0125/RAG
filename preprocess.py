import os
import shutil
import subprocess

def sync_and_convert_files(raw_dir, processed_dir, error_dir):
    """
    同步 01_raw_data 與 02_processed_data 檔案，同時比對修改時間與清理已刪除的檔案。
    如果有任何新增、更新或刪除，將回傳 True，更新向量資料庫。
    """
    print("\n--- 開始執行檔案前處理與更新檢查 ---")
    allowed_extensions = ['.pdf', '.docx']
    has_changes = False
    
    # 新增紀錄表，用來記錄所有應該存在的標準化檔案路徑
    expected_processed_files = set()
    
    # ==========================================
    # 第一階段：正向掃描 (檢查 01_raw_data 裡的資料是否有被 "新增" 或 "更新")
    # ==========================================
    for root, dirs, files in os.walk(raw_dir):
        for filename in files:
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
            
            elif ext == '.doc':
                processed_docx_path = os.path.join(target_dir, f"{name}.docx")
                # 將轉檔後的路徑記錄下來 (使用絕對路徑避免比對錯誤)
                expected_processed_files.add(os.path.abspath(processed_docx_path))
                
                if not os.path.exists(processed_docx_path) or raw_mtime > os.path.getmtime(processed_docx_path):
                    print(f"❗ 發現 .doc 檔案變動，正在重新轉檔: {display_name} ...")
                    try:
                        command = [
                            r"C:\Program Files\LibreOffice\program\soffice.exe", # 請確認指定執行檔路徑
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

    print("--- 檔案檢查完畢 ---\n")
    return has_changes