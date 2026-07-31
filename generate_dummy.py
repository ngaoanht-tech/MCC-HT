#!/usr/bin/env python3
"""生成占位训练数据，基于 2022 年的季节性统计特征 + 随机噪声"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
REF = BASE / "test" / "2022.csv"
SEED = 20260308
np.random.seed(SEED)

KEEP_REAL_TRAIN = [1959, 1961, 1962, 1963, 1964]
KEEP_REAL_TEST  = [2022]
KEEP_REAL_ALL   = KEEP_REAL_TRAIN + KEEP_REAL_TEST

# --- 读取参考数据，计算季节性统计 ---
ref = pd.read_csv(REF, encoding="utf-8-sig")
cols = list(ref.columns)
data_cols = [c for c in cols if c != "旬"]

stats = {}
for c in data_cols:
    vals = ref[c].values.astype(float)
    stats[c] = {"mean": vals.mean(), "std": max(vals.std(), vals.mean() * 0.01)}

def make_year(year, out_dir):
    rows = []
    for p in range(36):
        row = {"旬": str(p + 1)}
        for c in data_cols:
            s = stats[c]
            base = s["mean"] * np.random.uniform(0.7, 1.3)
            noise = np.random.normal(0, s["std"] * 0.3)
            row[c] = max(0.0, base + noise)
        rows.append(row)
    pd.DataFrame(rows)[cols].to_csv(Path(out_dir) / f"{year}.csv", index=False, encoding="utf-8-sig")

# --- 删除旧的 CSV（保留真实年份） ---
for dname in ["train", "test"]:
    for f in (BASE / dname).glob("*.csv"):
        if int(f.stem) not in KEEP_REAL_ALL:
            f.unlink()
            print(f"删除: {f.name}")

# --- 生成占位数据 ---
train_dummy = [1966,1967,1968,1969,1971,1972,1973,1974,1976,1977,1978,1979,1981,1982,1983,
               1984,1986,1987,1988,1989,1991,1992,1993,1994,1996,1997,1998,1999,2001,2002,
               2003,2004,2006,2007,2008,
               2009,2010,2011,2012,2013,2014]
for y in train_dummy:
    make_year(y, BASE / "train")
# test/ 只保留 2022，不生成任何占位测试数据

print(f"\n完成。保留真实年: train {KEEP_REAL_TRAIN} + test {KEEP_REAL_TEST}")
print(f"占位年: train {len(train_dummy)} 年 + test {len(test_dummy)} 年")
