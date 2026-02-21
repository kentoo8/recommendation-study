from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

RAW_DATA_DIR = Path("./data/raw")

# --- 1. データの読み込みと結合 ---
interactions_df = pd.read_csv(RAW_DATA_DIR / "interactions.csv")
companies_df = pd.read_csv(RAW_DATA_DIR / "companies.csv")
needs_df = pd.read_csv(RAW_DATA_DIR / "needs.csv")

# 学習用にSource企業の属性とTargetニーズの属性を結合
train_df = interactions_df.merge(
    companies_df, left_on="source_company_id", right_on="company_id"
)
train_df = train_df.merge(needs_df, left_on="target_needs_id", right_on="needs_id")

# TF Dataset化
interaction_ds = tf.data.Dataset.from_tensor_slices(
    {
        "source_id": train_df["source_company_id"].values.astype(str),
        "target_id": train_df["target_needs_id"].values.astype(str),
        "industry_l": train_df["industry_l"].values.astype(str),
        "needs_title": train_df["needs_title"].values.astype(str),
    }
)

# --- 2. ボキャブラリの作成 ---
source_id_lookup = keras.layers.StringLookup(
    vocabulary=companies_df["company_id"].unique().astype(str), mask_token=None
)
target_id_lookup = keras.layers.StringLookup(
    vocabulary=needs_df["needs_id"].unique().astype(str), mask_token=None
)

# 簡易的なテキストベクトル化（後ほどBERT等に差し替え可能ですわ）
text_vectorizer = keras.layers.TextVectorization()
# pandas 3.0のStringDtypeによるエラーを回避するため、標準のリスト(str型)に変換して適応しますわ
text_vectorizer.adapt(needs_df["needs_title"].astype(str).tolist())


# --- 3. モデルの定義 (Keras 3 ピュア実装) ---
class MatchingModel(keras.Model):
    def __init__(self):
        super().__init__()
        emb_dim = 32

        # Source Tower
        self.source_model = keras.Sequential(
            [
                source_id_lookup,
                keras.layers.Embedding(source_id_lookup.vocabulary_size(), emb_dim),
            ]
        )

        # Target Tower
        self.target_model = keras.Sequential(
            [
                target_id_lookup,
                keras.layers.Embedding(target_id_lookup.vocabulary_size(), emb_dim),
            ]
        )

    def call(self, inputs):
        # 推論時は単純にエンベディングを返すようにしておく
        return {
            "source_emb": self.source_model(inputs["source_id"]),
            "target_emb": self.target_model(inputs["target_id"]),
        }

    def train_step(self, data):
        # Keras 3 の標準的な学習ステップのオーバーライド
        features = data

        with tf.GradientTape() as tape:
            # 各タワーから特徴ベクトルを取得
            source_embeddings = self.source_model(features["source_id"])
            target_embeddings = self.target_model(features["target_id"])

            # In-batch Negative Sampling:
            # [batch_size, emb_dim]と[emb_dim, batch_size]の内積をとり [batch_size, batch_size] のスコア行列を作る
            logits = tf.matmul(source_embeddings, target_embeddings, transpose_b=True)

            # 対角成分が正例（あるsourceに対する正しいtarget）となる
            batch_size = tf.shape(logits)[0]
            labels = tf.range(batch_size)

            # Sparse Categorical Crossentropyで最適化
            loss = keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            )

            # バッチ全体の平均損失
            loss = tf.reduce_mean(loss)

            # In-batch Accuracy (自身を正しくトップ1に当てられたか)
            predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
            accuracy = tf.reduce_mean(tf.cast(predictions == labels, tf.float32))

        # 勾配の計算と適用
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss, "in_batch_accuracy": accuracy}


# --- 4. 学習の開始 ---
model = MatchingModel()
model.compile(optimizer=keras.optimizers.Adagrad(0.1))

print("Keras 3 ピュア実装にて学習を開始いたしますわ...")
# バッチサイズ分だけ負例サンプリングができるため、バッチサイズは少し大きめが良いですわ
model.fit(interaction_ds.batch(128), epochs=5)
print("本日の目標、純粋なKeras 3での学習の実行に成功いたしましたわ！")
