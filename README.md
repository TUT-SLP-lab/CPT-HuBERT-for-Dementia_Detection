# CPT-HuBERT-for-Dementia_Detection
This is the code for continuing HuBERT's Continual Pre-Training.

Hugging Face `transformers`ライブラリをベースとし、HuBERTの論文で提案されているk-meansによる擬似ラベル生成とマスク化予測の2段階プロセスを再現しています。

## 学習プロセス

学習は、4つのステップを順番に実行することで行います。

### ステップ0：データ準備

- 学習に使用したい音声ファイル（`.wav`など）を一つのフォルダにまとめます。
- `metadata.csv`を作成します。このCSVには`file_path`というヘッダーを持つ列を1つだけ用意し、各行に音声ファイルへのフルパスを記述します。

### ステップ1：特徴抽出

音声波形を、クラスタリング可能な特徴量ベクトルに変換します。

```bash
python 1_extract_feature.py
```
実行後、`hubert_features/`フォルダに多数の`.npy`ファイルが生成されます。

### ステップ2：K-meansクラスタリング

抽出した特徴量をクラスタリングし、学習の「正解ラベル」となる擬似ラベルを生成します。

```bash
python 2_k_means.py
```
実行後、`hubert_labels/`フォルダに多数の`.labels.npy`ファイルが生成されます。

### ステップ3：メタデータの更新

`metadata.csv`に、生成したラベルファイルへのパスを追加します。

```bash
python step3_update_metadata.py
```
実行後、`metadata_updated.csv`が作成されるので、古い`metadata.csv`を削除し、このファイル名を`metadata.csv`に変更してください。

### ステップ4：追加事前学習の実行

準備したデータとラベルを使って、実際の学習を開始します。

```bash
python HuBERT_CPT.py
```
学習が開始され、進捗が`wandb`で確認できます。完了後、学習済みモデルが`output_dir`（デフォルトでは`./hubert-base-japanese-continued-pretraining`）に保存されます。

---
## 各スクリプトの役割

- **`HuBERT_CPT.py`**:
    データの前処理、カスタムモデルの定義、`Trainer`を使った学習の実行までを行います。

- **`1_extract_feature.py`**:
    **[ステップ1]** `rinna/japanese-hubert-base`を教師モデルとして使用し、データセット内の全音声から中間層の特徴量を抽出して保存します。

- **`2_k_means.py`**:
    **[ステップ2]** 抽出された全特徴量に対して`faiss`でk-meansクラスタリングを行い、各特徴量に対応する擬似ラベル（クラスタID）を生成・保存します。

- **`step3_update_metadata.py`**:
    **[ステップ3]** `metadata.csv`に、各音声に対応するラベルファイルのパスを追記するためのスクリプトです。
