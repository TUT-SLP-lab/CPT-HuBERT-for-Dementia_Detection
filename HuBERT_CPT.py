import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os
import datasets
import soundfile as sf
import numpy as np
from datasets import load_dataset
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import (
    HubertModel,
    HubertPreTrainedModel,
    HubertConfig,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import CausalLMOutput

# 音声デコーダーとしてtorchcodecの代わりにsoundfileを使うよう指定
datasets.config.AUDIO_DECODER = "soundfile"

class CustomHuBERTForPreTraining(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.projection_head = nn.Linear(config.hidden_size, config.num_clusters)
        self.init_weights()

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 内部特徴量の抽出と次元入れ替え
        hidden_states = self.hubert.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.hubert.feature_projection(hidden_states)

        if attention_mask is not None:
            attention_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )

        # マスキング処理
        batch_size, sequence_length, hidden_size = hidden_states.shape
        device = hidden_states.device

        # マスクする位置を決定するためのブール値マスクを作成
        mask_time_indices = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)
        
        for i in range(batch_size):
            # マスクするフレーム数を計算
            num_frames_to_mask = int(sequence_length * self.config.mask_time_prob)
            # マスクを開始する位置をランダムに選択
            mask_starts = torch.randperm(sequence_length - self.config.mask_time_length + 1, device=device)[:num_frames_to_mask]
            
            for start in mask_starts:
                end = start + self.config.mask_time_length
                mask_time_indices[i, start:end] = True
        
        # マスクを適用
        masked_hidden_states = hidden_states.clone()

        # マスクされた部分を、学習済みの特別なベクトルで置き換える
        masked_hidden_states[mask_time_indices] = self.hubert.masked_spec_embed.to(hidden_states.dtype)

        # マスクされた特徴量をTransformerに通す
        encoder_outputs = self.hubert.encoder(
            masked_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 損失計算など
        if isinstance(encoder_outputs, dict):
            prediction_states = encoder_outputs.last_hidden_state
        else:
            prediction_states = encoder_outputs[0]

        logits = self.projection_head(prediction_states[mask_time_indices])
        
        loss = None
        if labels is not None:
            masked_labels = labels[mask_time_indices]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_clusters), masked_labels.view(-1))

        if not return_dict:
            if isinstance(encoder_outputs, dict):
                output = (logits,) + tuple(v for k, v in encoder_outputs.items() if k != "last_hidden_state")
            else:
                output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

@dataclass
class DataCollatorForHubertPretraining:

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # input_valuesのバッチ作成
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt", return_attention_mask=True
        )

        # labelsのシーケンスをリストとして取り出す
        label_sequences = [feature["labels"] for feature in features]
        
        # このバッチ内でのラベルの最大長を計算
        max_label_length = max(len(seq) for seq in label_sequences)
        
        # -100で埋めたテンソルを作成
        padded_labels = torch.full((len(label_sequences), max_label_length), fill_value=-100, dtype=torch.long)
        
        # 各ラベルシーケンスを、パディング済みテンソルにコピー
        for i, seq in enumerate(label_sequences):
            padded_labels[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            
        batch["labels"] = padded_labels

        print("--- DataCollatorの最終出力シェイプ ---")
        print(f"input_values: {batch['input_values'].shape}, labels: {batch['labels'].shape}")
        print("------------------------------------")

        return batch

# wandbの設定
os.environ["WANDB_PROJECT"] = "CPT-rinna-japanese-hubert-base"

model_name = "rinna/japanese-hubert-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

config = HubertConfig.from_pretrained(model_name)

print("マスキング用のConfigパラメータを設定します...")
config.mask_time_prob = 0.05         # マスクを開始する確率
config.mask_time_length = 10         # マスクする長さ
config.mask_feature_prob = 0.0       # 特徴量方向のマスキング確率
config.mask_feature_length = 10      # 特徴量方向のマスキング長

config.num_clusters = 100 

model = CustomHuBERTForPreTraining.from_pretrained(model_name, config=config)

dataset = load_dataset("csv", data_files={"train": "./metadata.csv"})

def preprocess_function(examples):

    audio_arrays = []
    label_arrays = []
    
    # file_pathとlabel_pathを同時にループ処理
    for audio_path, label_path in zip(examples["file_path"], examples["label_path"]):
        # 音声の読み込み
        speech_array, sampling_rate = sf.read(audio_path)
        
        # ラベルの読み込み
        label_array = np.load(label_path)

        # 空ファイル対策
        if len(speech_array) == 0:
            print(f"音声ファイルが空です。スキップします: {audio_path}")
            continue 

        audio_arrays.append(speech_array)
        label_arrays.append(label_array)

    # 音声をモデル入力形式に変換（切り詰め＆パディング）
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=246000
    )
    
    # ラベルも音声と同じ長さに合わせて調整する
    # モデルのダウンサンプリング率を考慮
    max_feature_length = inputs.input_values.shape[1] // 320
    
    processed_labels = []
    for l_arr in label_arrays:
        # ラベルも同じように切り詰める
        processed_labels.append(l_arr[:max_feature_length])
        
    # DataCollatorに渡すために、labelsをinputs辞書に追加
    inputs["labels"] = processed_labels
    return inputs


processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=dataset["train"].column_names # 元の列をすべて削除
)

# 以前のDataCollatorWithPaddingの代わりに、新しいCollatorを使う
data_collator = DataCollatorForHubertPretraining(
    feature_extractor=feature_extractor,
    padding="longest",
)

training_args = TrainingArguments(
    output_dir="./hubert-base-japanese-continued-pretraining",
    run_name="hubert-pretrain-run-2", 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=500,
    max_steps=5000,
    fp16=True,
    save_steps=500,
    logging_steps=50,
    report_to="wandb",
)

# 5. 学習の実行（変更なし）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# 学習済みモデルの保存
trainer.save_model("./hubert-base-japanese-continued-pretraining/final_model-1")
feature_extractor.save_pretrained("./hubert-base-japanese-continued-pretraining/final_model-1")

# wandbのセッションを終了
import wandb
wandb.finish()