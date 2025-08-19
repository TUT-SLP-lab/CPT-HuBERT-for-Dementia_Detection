import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoFeatureExtractor,
    HubertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

class SpeechDataset(Dataset):
    def __init__(self, csv_path, feature_extractor, label2id):
        df = pd.read_csv(csv_path)
        self.paths = df["path"].tolist()
        # ラベル文字列 → 整数
        classes = sorted(df["Class"].unique())
        self.label2id = {c:i for i,c in enumerate(classes)}
        self.labels = [self.label2id[c] for c in df["Class"]]
        self.fe = feature_extractor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, _ = librosa.load(self.paths[idx], sr=16000)
        inputs = self.fe(wav, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"].squeeze(0)
        attention_mask = torch.ones_like(input_values, dtype=torch.long)
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall":    recall_score(labels, preds, average="weighted"),
        "f1":       f1_score(labels, preds, average="weighted")
    }

def main():
    # 1) Feature Extractor とモデル
    fe    = AutoFeatureExtractor.from_pretrained("rinna/japanese-hubert-base")
    model = HubertForSequenceClassification.from_pretrained(
        "rinna/japanese-hubert-base", num_labels=2
    )

    # 2) データセット
    label2id = {"CN":0, "pAD":1}
    train_ds = SpeechDataset("data/train.csv", fe, label2id)
    val_ds   = SpeechDataset("data/val.csv",   fe, label2id)

    # 3) data_collator を fe クロージャで定義
    def data_collator(features):
        # 1) バッチ内で最大シーケンス長を取得
        max_len = max(f["input_values"].shape[0] for f in features)

        batch_inputs = []
        batch_masks  = []
        batch_labels = []

        for f in features:
            iv = f["input_values"]     # Tensor of shape (L,)
            am = f["attention_mask"]   # Tensor of shape (L,)
            lbl = f["labels"]          # Tensor scalar

            pad_len = max_len - iv.shape[0]
            # 2) 後ろにゼロパディング
            iv_padded = torch.cat([iv, torch.zeros(pad_len, dtype=iv.dtype)])
            am_padded = torch.cat([am, torch.zeros(pad_len, dtype=am.dtype)])

            batch_inputs.append(iv_padded)
            batch_masks.append(am_padded)
            batch_labels.append(lbl)

        # 3) ミニバッチ化
        batch = {
            "input_values":    torch.stack(batch_inputs),   # (B, max_len)
            "attention_mask":  torch.stack(batch_masks),    # (B, max_len)
            "labels":          torch.stack(batch_labels),   # (B,)
        }
        return batch

    # 4) TrainingArguments

    args = TrainingArguments(
        output_dir="checkpoints",          # チェックポイント保存先
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8, 
        num_train_epochs=3,                # エポック数
        learning_rate=1e-5,                # 学習率
        eval_strategy="steps",             # ステップごとに eval
        save_strategy="steps",             # ステップごとに checkpoint
        eval_steps=50,                     # 100step ごとにバリデーション
        save_steps=50,                     # ←eval_steps と合わせる
        logging_steps=50,                  # ロス等のログは 50step ごと
        load_best_model_at_end=True,       # 学習後にベストモデルを自動で読み込む
        metric_for_best_model="eval_loss", # ベストモデル判定は eval_loss が最小のもの
        greater_is_better=False,           # ロスは小さいほど良いので False
        save_total_limit=3,                # チェックポイントは直近3つまでキープ
        fp16=True,                         # GPU が対応していれば混合精度で高速化
        report_to="wandb",
        run_name="FT-rinna-hubert-base",
        logging_dir="logs/ft_wandb",
    )


    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6) 学習スタート
    trainer.train()
    trainer.save_model("final_model")  # 最終モデルを保存

    # 7) テストセットで最終評価
    test_ds = SpeechDataset("data/test.csv", fe, label2id)
    metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print("Test set metrics:", metrics)

if __name__ == "__main__":
    main()
