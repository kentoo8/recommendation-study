# Two-Tower モデル `train_step` の数理的解説

## 概要

本プロジェクトの推薦モデルは **Two-Tower（双塔）アーキテクチャ** を採用しており、
`train_step` メソッド内で **In-batch Negative Sampling（バッチ内負例サンプリング）** による対照学習を行っています。

---

## 1. 入力と埋め込み（Embedding）

ミニバッチ内の $B$ 個のインタラクションデータについて、Source（企業）と Target（ニーズ）それぞれを $d$ 次元のベクトル空間に写像します。

$$
s_i = f_{\theta}(\text{source}_i) \in \mathbb{R}^d, \quad i = 1, \dots, B
$$

$$
t_j = g_{\phi}(\text{target}_j) \in \mathbb{R}^d, \quad j = 1, \dots, B
$$

- $\text{source}_i$ : バッチ内 $i$ 番目のサンプルの企業 ID（文字列）。`StringLookup` で整数インデックスに変換されて Embedding 層に渡されます。
- $\text{target}_j$ : バッチ内 $j$ 番目のサンプルのニーズ ID（文字列）。同様に整数インデックス経由で Embedding 層に渡されます。
- $f_{\theta}$ : Source Tower（企業側の Embedding 関数、パラメータ $\theta$ ）
- $g_{\phi}$ : Target Tower（ニーズ側の Embedding 関数、パラメータ $\phi$ ）

---

## 2. スコア行列の計算（内積）

バッチ内のすべての $(i, j)$ のペアについて、両タワーの出力ベクトルの内積を計算し、 $B \times B$ のスコア行列 $L$ を構成します。

$$
L_{ij} = s_i^\top t_j
$$

行列形式で表すと: 

$$
L = S \, T^\top \in \mathbb{R}^{B \times B}
$$

ここで $S = [s_1, \dots, s_B]^\top \in \mathbb{R}^{B \times d}$, $T = [t_1, \dots, t_B]^\top \in \mathbb{R}^{B \times d}$ です。

対応するコード: 

```python
logits = tf.matmul(source_embeddings, target_embeddings, transpose_b=True)
```

---

## 3. 正例と負例の定義（In-batch Negative Sampling）

元のデータにおいて、$(\text{source}_i, \text{target}_i)$ は実際に観測されたペア（正例）です。
したがって、スコア行列 $L$ の**対角成分** $L_{ii}$ が正例のスコアに該当します。

$$
\text{正例のスコア: } \quad L_{ii} = s_i^\top t_i
$$

$$
\text{負例のスコア: } \quad L_{ij} = s_i^\top t_j \quad (j \neq i)
$$

つまり、各サンプル $i$ にとって、同じバッチ内の**他の $B - 1$ 個のターゲット**がすべて負例として再利用されます。
これが **In-batch Negative Sampling** の核心です。

対応するコード: 

```python
labels = tf.range(batch_size)  # → [0, 1, 2, ..., B-1]
```

---

## 4. 損失関数（Softmax Cross-Entropy）

各サンプル $i$ について、$B$ クラスの分類問題として Softmax Cross-Entropy Loss を計算します。
正解ラベルは $y_i = i$ （自分自身のインデックス）です。

$$
\mathcal{L}_i = -\log \frac{\exp(L_{ii})}{\displaystyle\sum_{j=1}^{B} \exp(L_{ij})}
$$

バッチ全体の平均損失: 

$$
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_i
$$

対応するコード: 

```python
loss = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
loss = tf.reduce_mean(loss)
```

---

## 5. 評価指標（In-batch Accuracy）

各サンプル $i$ について、スコア行列の $i$ 行目で最もスコアの高いインデックスが $i$ 自身であるかを判定します。

$$
\hat{y}_i = \arg\max_{j} L_{ij}
$$

$$
\text{Accuracy} = \frac{1}{B} \sum_{i=1}^{B} \mathbb{1}[\hat{y}_i = i]
$$

対応するコード: 

```python
predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
accuracy = tf.reduce_mean(tf.cast(predictions == labels, tf.float32))
```

---

## 6. パラメータの更新

上記の損失 $\mathcal{L}$ に対し、自動微分で勾配を求め、Adagrad で更新します。

$$
\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}, \qquad
\phi \leftarrow \phi - \eta \cdot \frac{\partial \mathcal{L}}{\partial \phi}
$$

対応するコード: 

```python
gradients = tape.gradient(loss, self.trainable_variables)
self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
```

---

## 全体の流れ（図解）

```
                  入力バッチ (B個のペア)
                            │
                 ┌──────────┴──────────┐
                 │                     │
           Source Tower          Target Tower
                 │                     │
                 ↓                     ↓
           S ∈ ℝ^{B×d}          T ∈ ℝ^{B×d}
                 │                     │
                 └──────────┬──────────┘
                            │
                  L = S・Tᵀ ∈ ℝ^{B×B}    ← 内積スコア行列
                            │
                labels = [0, 1, ..., B-1]  ← 対角が正解
                            │
                Softmax Cross-Entropy Loss
                            │
                     勾配計算 & 更新
```

---

## 7. 具体的な数値例（ $B=5,\ d=3$ ）

### ステップ0: 入力バッチ

バッチサイズ $B=5$ の場合、以下のような 5 組のインタラクション（企業ID, ニーズID）が入力されますわ。

| $i$ | source\_id（企業） | target\_id（ニーズ） |
|:---:|:---:|:---:|
| 0 | `company_03` | `needs_07` |
| 1 | `company_11` | `needs_02` |
| 2 | `company_07` | `needs_15` |
| 3 | `company_19` | `needs_09` |
| 4 | `company_05` | `needs_12` |

---

### ステップ1: Embedding テーブルによる変換（次元数 $d=3$）

各 ID はまず整数インデックスに変換（`StringLookup`）され、その後 Embedding テーブルから対応する $d=3$ 次元ベクトルが取り出されますわ。

$$
\underbrace{\text{company03}}_{\text{ID}} \xrightarrow{\text{StringLookup}} \underbrace{3}_{\text{index}} \xrightarrow{\text{Embedding}} \underbrace{(0.9,\ 0.1,\ 0.3)}_{\text{ベクトル} \in \mathbb{R}^3}
$$

5 件分を縦に並べると、それぞれ $5 \times 3$ の行列 $S$ （Source）と $T$ （Target）になりますの。

$$
S =
\begin{pmatrix}
0.9 & 0.1 & 0.3 \\
0.2 & 0.8 & 0.1 \\
0.5 & 0.5 & 0.7 \\
0.1 & 0.3 & 0.9 \\
0.7 & 0.2 & 0.4
\end{pmatrix}, \qquad
T =
\begin{pmatrix}
0.8 & 0.2 & 0.4 \\
0.3 & 0.7 & 0.2 \\
0.4 & 0.6 & 0.6 \\
0.2 & 0.1 & 0.8 \\
0.6 & 0.3 & 0.5
\end{pmatrix}
$$

各行が1サンプルのベクトルですわ（ $i$ 行目が第 $i$ 番目の企業／ニーズ）。


---

### ステップ2: スコア行列 $L = ST^\top$ （ $5 \times 5$ ）

$$
L =
\begin{pmatrix}
\boxed{0.86} & 0.41 & 0.60 & 0.44 & 0.71 \\
0.51 & \boxed{0.64} & 0.62 & 0.21 & 0.49 \\
0.82 & 0.63 & \boxed{1.12} & 0.69 & 0.89 \\
0.44 & 0.24 & 0.68 & \boxed{0.77} & 0.51 \\
0.78 & 0.43 & 0.82 & 0.47 & \boxed{0.74}
\end{pmatrix}
$$

**太字**が正例（対角成分）のスコアですわ。各行において、対角成分が最大になっていれば「正しく推薦できている」状態でございます。

---

### ステップ3: 正解ラベルと損失計算

正解ラベルは: 

$$
\boldsymbol{y} = [0,\ 1,\ 2,\ 3,\ 4]
$$

第1サンプル（ $i=0$ ）の Softmax Cross-Entropy: 

$$
\mathcal{L}_0 = -\log \frac{e^{0.86}}{e^{0.86} + e^{0.41} + e^{0.60} + e^{0.44} + e^{0.71}}
$$

$$
= -\log \frac{2.363}{9.279} = -\log(0.255) \approx 1.37
$$

同様に全サンプルの損失を計算し、平均を取ったものがバッチ全体の損失 $\mathcal{L}$ になりますの。

---

### ステップ4: In-batch Accuracy の確認

各行の $\arg\max$: 

| サンプル $i$ | スコア行の最大値の列 | 正解 $i$ | 正解？ |
|:---:|:---:|:---:|:---:|
| 0 | 0（スコア 0.86） | 0 | ✅ |
| 1 | 2（スコア 0.62） | 1 | ❌ |
| 2 | 2（スコア 1.12） | 2 | ✅ |
| 3 | 3（スコア 0.77） | 3 | ✅ |
| 4 | 2（スコア 0.82） | 4 | ❌ |

$$
\text{In-batch Accuracy} = \frac{3}{5} = 0.60
$$

3件正解、2件はまだ学習が必要な状態ですわね。損失を小さくするよう学習が進むにつれ、対角成分のスコアが他の成分よりも大きくなっていき、Accuracy が上がっていくのが理想の姿ですわ。
