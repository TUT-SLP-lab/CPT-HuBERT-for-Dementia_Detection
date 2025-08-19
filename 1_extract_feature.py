import torch
import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from pathlib import Path
import os
from tqdm import tqdm

# --- ユーザー設定 ---
METADATA_FILE = "./metadata.csv"
FEATURE_OUTPUT_DIR = "hubert_features"
MODEL_NAME = "rinna/japanese-hubert-base"
EXTRACTION_LAYER = 9
MAX_DURATION_S = 30


def extract_features():
    """
    長い音声を分割しながら、全音声からHuBERTの特徴量を抽出し、ファイルに保存する。
    """
    # 1. セットアップ (変更なし)
    print("--- セットアップ開始 ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    
    output_dir = Path(FEATURE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"特徴量の保存先: {output_dir.resolve()}")

    # 2. モデルとプロセッサのロード (変更なし)
    print(f"モデル '{MODEL_NAME}' をロード中...")
    model = HubertModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    
    # 3. データセットのロード (変更なし)
    print(f"データセット '{METADATA_FILE}' をロード中...")
    dataset = load_dataset("csv", data_files={"train": METADATA_FILE})["train"]
    
    total_files = len(dataset)
    print(f"合計 {total_files} 個のファイルを処理します。")

    # 4. 1ファイルずつ、分割しながら特徴抽出
    with torch.no_grad():
        for example in tqdm(dataset, desc="特徴抽出中"):
            audio_path = example["file_path"]
            
            # 音声ファイルを読み込み
            audio_array, sampling_rate = sf.read(audio_path)
            
            # 分割処理
            max_length = int(MAX_DURATION_S * sampling_rate)
            chunks = [audio_array[i : i + max_length] for i in range(0, len(audio_array), max_length)]
            
            all_features = []
            for chunk in chunks:
                # 各チャンクをモデルへの入力形式に変換
                inputs = feature_extractor(
                    [chunk],
                    sampling_rate=sampling_rate,
                    return_tensors="pt"
                ).to(device)

                # モデルを実行
                outputs = model(**inputs, output_hidden_states=True)
                
                # 指定した層の隠れ状態を取得
                hidden_states = outputs.hidden_states[EXTRACTION_LAYER].squeeze(0)
                all_features.append(hidden_states.cpu().numpy())

            # 分割された特徴量を結合して1つのファイルにする
            feature_vector = np.concatenate(all_features, axis=0)

            # 特徴量をファイルに保存
            original_path = Path(audio_path)
            output_path = output_dir / f"{original_path.stem}.npy"
            np.save(output_path, feature_vector)

    print("\n特徴抽出が完了しました")
    print(f"'{FEATURE_OUTPUT_DIR}' フォルダに {total_files} 個の.npyファイルが保存されました")


if __name__ == "__main__":
    extract_features()