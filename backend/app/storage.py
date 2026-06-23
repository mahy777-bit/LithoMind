import os
from huggingface_hub import snapshot_download
from app.config import HF_REPO_ID, HF_TOKEN, LOCAL_DB_PATH

def db_exists_locally() -> bool:
    return os.path.exists(LOCAL_DB_PATH) and os.listdir(LOCAL_DB_PATH)

def download_db() -> bool:
    print("Downloading knowledge base from Hugging Face...")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DB_PATH,
            token=HF_TOKEN
        )
        print("Knowledge base ready.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def sync_db() -> bool:
    if db_exists_locally():
        print("Local knowledge base found — ready.")
        return True
    return download_db()

def upload_db() -> bool:
    print("Uploading knowledge base to Hugging Face...")
    try:
        from huggingface_hub import upload_folder
        upload_folder(
            folder_path=LOCAL_DB_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("Upload successful.")
        return True
    except Exception as e:
        print(f"Upload failed: {e}")
        return False