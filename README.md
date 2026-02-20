# Recommendation Study

BtoBマッチングプラットフォームにおける、企業とニーズの推薦（レコメンデーション）システムの検証用プロトタイプでございます。
TensorFlow Recommenders (TFRS) ではなく、**Keras 3 ピュアな実装（カスタム Two-Tower モデル）**で構築されておりますの。

## 使い方

```bash
# uv で依存関係をインストール
uv sync

# サンプルデータの生成
uv run generate_sample_data.py

# 学習の実行
uv run training.py
```
