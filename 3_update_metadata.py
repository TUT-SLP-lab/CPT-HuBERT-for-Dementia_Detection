import pandas as pd
from pathlib import Path

# --- ユーザー設定 ---
METADATA_FILE = "./metadata.csv"
LABEL_DIR = "hubert_labels"

print(f"'{METADATA_FILE}'を更新中")
df = pd.read_csv(METADATA_FILE)

# ラベルファイルへのフルパスを新しい列として追加
label_dir_path = Path(LABEL_DIR).resolve()
df["label_path"] = df["file_path"].apply(
    lambda x: str(label_dir_path / f"{Path(x).stem}.labels.npy")
)

df.to_csv("metadata_updated.csv", index=False)
print(f" 'metadata_updated.csv' が作成されました。")
