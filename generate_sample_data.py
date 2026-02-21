import random
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("./data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

NUM_COMPANIES = 100
NUM_NEEDS = 150
NUM_INTERACTIONS = 300

# 1. Companies CSV
industries = [
    ("情報通信業", ["ソフトウェア", "AI開発", "インフラ構築", "システム運用"]),
    ("製造業", ["金属製品", "精密加工", "自動車部品", "食品加工"]),
    ("卸売業", ["産業機械", "電子部品", "建築資材", "アパレル"]),
    ("サービス業", ["コンサルティング", "人材派遣", "広告代理店", "デザイン制作"]),
]

company_features = [
    ("クラウド開発に強み", "営業不足"),
    ("技術承継", "後継者不足"),
    ("海外製品の輸入販売", "保守エンジニア不足"),
    ("独自技術あり", "資金不足"),
    ("若手が多い", "マネジメント層不足"),
    ("老舗企業", "デジタル化の遅れ"),
]

companies_data = []
for i in range(1, NUM_COMPANIES + 1):
    c_id = f"c{i:03d}"
    c_name = f"サンプル企業{i}株式会社"
    ind_l, ind_m_list = random.choice(industries)
    ind_m = random.choice(ind_m_list)
    ind_s = f"{ind_m}関連事業"
    desc, challenge = random.choice(company_features)
    companies_data.append([c_id, c_name, ind_l, ind_m, ind_s, desc, challenge])

pd.DataFrame(
    companies_data,
    columns=[
        "company_id",
        "company_name",
        "industry_l",
        "industry_m",
        "industry_s",
        "description",
        "challenges",
    ],
).to_csv(RAW_DATA_DIR / "companies.csv", index=False)

# 2. Needs CSV
needs_titles = [
    "パートナー開拓",
    "AI教育支援",
    "自動化の相談",
    "技術者募集",
    "システム刷新",
    "コスト削減提案",
    "新規事業開発",
    "マーケティング支援",
]
needs_details = [
    "代理店求む",
    "研修提供希望",
    "ロボット導入検討中",
    "保守パートナー求む",
    "レガシーシステムからの移行",
    "業務効率化のツール導入",
    "協業パートナー募集",
    "Web集客の強化",
]

needs_data = []
for i in range(1, NUM_NEEDS + 1):
    n_id = f"n{i:03d}"
    t_c_id = f"c{random.randint(1, NUM_COMPANIES):03d}"
    title = random.choice(needs_titles)
    detail = random.choice(needs_details)
    needs_data.append([n_id, t_c_id, title, detail])

pd.DataFrame(
    needs_data, columns=["needs_id", "target_company_id", "needs_title", "needs_detail"]
).to_csv(RAW_DATA_DIR / "needs.csv", index=False)

# 3. Interactions CSV
event_types = [("view", 1), ("request", 5), ("accept", 10), ("close", 50)]

interactions_data = []
for _ in range(NUM_INTERACTIONS):
    s_c_id = f"c{random.randint(1, NUM_COMPANIES):03d}"
    # pick a random need
    need = random.choice(needs_data)
    t_n_id = need[0]
    t_c_id = need[1]

    # ensure source isn't same as target
    if s_c_id == t_c_id:
        continue

    e_type, weight = random.choice(event_types)
    interactions_data.append([s_c_id, t_n_id, t_c_id, e_type, weight])

pd.DataFrame(
    interactions_data,
    columns=[
        "source_company_id",
        "target_needs_id",
        "target_company_id",
        "event_type",
        "weight",
    ],
).to_csv(RAW_DATA_DIR / "interactions.csv", index=False)

print(
    f"{len(companies_data)}件の企業データ、{len(needs_data)}件のニーズデータ、{len(interactions_data)}件のインタラクションデータの作成が完了いたしましたわ。"
)
