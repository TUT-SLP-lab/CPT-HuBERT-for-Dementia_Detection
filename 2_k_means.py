import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
import os

# --- ユーザー設定 ---
# ステップ1で特徴量を保存したフォルダ
FEATURE_DIR = "hubert_features"

# 生成したラベルを保存するフォルダ名
LABEL_OUTPUT_DIR = "hubert_labels"

# クラスタの数 (学習スクリプトの config.num_clusters と同じ値にする)
NUM_CLUSTERS = 100

# K-meansの学習に使う特徴量の割合 (多すぎるとメモリ不足になる)
# 0.1 = 10%のデータで代表点を学習し、全データにラベルを割り当て
SUBSAMPLE_PERCENT = 0.1
# --- 設定はここまで ---

def run_kmeans():
    """
    抽出された特徴量に対してk-meansクラスタリングを実行し、ラベルを生成する。
    """
    print("--- K-meansクラスタリング開始 ---")
    feature_dir = Path(FEATURE_DIR)
    label_dir = Path(LABEL_OUTPUT_DIR)
    label_dir.mkdir(parents=True, exist_ok=True)
    print(f"特徴量の読み込み元: {feature_dir.resolve()}")
    print(f"ラベルの保存先: {label_dir.resolve()}")

    # 1. 特徴量の読み込みとサブサンプリング
    print("\nステップ1/3: 特徴量を読み込み、学習用にサンプリング中")
    all_feature_paths = list(feature_dir.glob("*.npy"))
    
    # まずは学習に使うデータだけをサンプリングしてメモリにロード
    training_features = []
    for path in tqdm(all_feature_paths, desc="サンプリング中"):
        features = np.load(path)
        if len(features) > 0:
            num_samples = int(len(features) * SUBSAMPLE_PERCENT)
            if num_samples == 0 and len(features) > 0:
                num_samples = 1 # 少なくとも1つはサンプリング
            
            sample_indices = np.random.choice(len(features), num_samples, replace=False)
            training_features.append(features[sample_indices])
    
    training_features = np.concatenate(training_features, axis=0)
    print(f"合計 {training_features.shape[0]} 個の特徴量ベクトルでK-meansモデルを訓練")
    
    # 2. FaissでK-meansモデルを訓練
    print("\nステップ2/3: K-meansモデルの訓練")
    n_centroids = NUM_CLUSTERS
    d = training_features.shape[1] # 特徴量の次元数 (768)
    
    kmeans = faiss.Kmeans(d=d, k=n_centroids, niter=20, verbose=True, gpu=True)
    kmeans.train(training_features.astype(np.float32))
    print("K-meansモデルの訓練が完了しました。")

    # メモリを解放
    del training_features

    # 3. 全データにラベルを割り当てて保存
    print("\nステップ3/3: 全データにラベルを割り当てて保存中")
    for path in tqdm(all_feature_paths, desc="ラベル割り当て中"):
        features = np.load(path).astype(np.float32)
        if len(features) > 0:
            # 各特徴量がどのクラスタに最も近いかを計算し、そのIDを取得
            _, labels = kmeans.index.search(features, 1)
            labels = labels.squeeze(1) # (N, 1) -> (N,)
            
            # ラベルを保存
            output_path = label_dir / f"{path.stem}.labels.npy"
            np.save(output_path, labels)

    print("\nK-meansクラスタリングが完了しました")
    print(f"'{LABEL_OUTPUT_DIR}' フォルダに {len(all_feature_paths)} 個の.labels.npyファイルが保存されました。")


if __name__ == "__main__":
    run_kmeans()