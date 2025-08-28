#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbo出力のグローバル重複統合（複数日の first_seen + embeddings をマージ）

入力: --inputs で複数ディレクトリ（各ディレクトリに first_seen.csv と 任意で turbo_embeddings.npz）
出力: --outdir に以下を保存
  - merged_first_seen.csv: cluster_id, first_ts（最初に観測された時刻）
  - mapping.csv: key(old) -> cluster_id（keyは "{camera_id}_{track_id}"）
  - clusters.csv: cluster_id, members（カンマ区切り）
  - stats.csv: 件数等の要約

注意:
- embeddings が全入力から見つからない場合は重複統合なし（単純結合）になります。
"""
import os
import csv
import argparse
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np


def parse_iso(s: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return dt.datetime(1970, 1, 1)


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def load_first_seen(dirpath: str) -> Dict[str, str]:
    path = os.path.join(dirpath, "first_seen.csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            # columns: camera_id, track_id, first_ts
            cam, tid, ts = row[0], row[1], row[2]
            key = f"{cam}_{tid}"
            out[key] = ts
    return out


def load_embeddings(dirpath: str) -> Dict[str, np.ndarray]:
    path = os.path.join(dirpath, "turbo_embeddings.npz")
    out: Dict[str, np.ndarray] = {}
    if not os.path.isfile(path):
        return out
    try:
        z = np.load(path)
        for k in z.files:
            v = z[k]
            if isinstance(v, np.ndarray):
                vv = v.astype(np.float32)
                n = np.linalg.norm(vv) + 1e-9
                out[k] = (vv / n).astype(np.float32)
    except Exception:
        pass
    return out


def cluster_by_threshold(keys: List[str], vecs: np.ndarray, thr: float) -> List[int]:
    n = len(keys)
    if n == 0:
        return []
    dsu = DSU(n)
    # 正規化済み前提。ブロックごとに内積を計算
    blk = 2048
    for i0 in range(0, n, blk):
        i1 = min(n, i0 + blk)
        A = vecs[i0:i1]
        # 対角（同一要素）は無視する
        # 下三角も含めて重複unionしないようjブロックはiブロックから開始
        for j0 in range(i0, n, blk):
            j1 = min(n, j0 + blk)
            B = vecs[j0:j1]
            sims = A @ B.T  # (i1-i0, j1-j0)
            ii, jj = np.where(sims >= thr)
            for u, v in zip(ii.tolist(), jj.tolist()):
                a = i0 + u
                b = j0 + v
                if a == b:
                    continue
                dsu.union(a, b)
    # 代表インデックスへ圧縮
    reps = [dsu.find(i) for i in range(n)]
    # 連番cluster id
    rep_to_cluster: Dict[int, int] = {}
    clusters: List[int] = [0] * n
    cur = 0
    for i, r in enumerate(reps):
        if r not in rep_to_cluster:
            rep_to_cluster[r] = cur
            cur += 1
        clusters[i] = rep_to_cluster[r]
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="first_seen/turbo_embeddings を含むディレクトリ群")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--alias-threshold", type=float, default=0.92)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 収集
    all_first: Dict[str, str] = {}
    all_vecs: Dict[str, np.ndarray] = {}
    for d in args.inputs:
        fs = load_first_seen(d)
        emb = load_embeddings(d)
        all_first.update(fs)
        all_vecs.update(emb)

    keys = list(all_first.keys())
    # embeddings があるキーだけクラスタリング対象
    emb_keys = [k for k in keys if k in all_vecs]
    if len(emb_keys) == 0:
        # 重複統合なしで単純出力
        out_path = os.path.join(args.outdir, "merged_first_seen.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cluster_id", "first_ts", "members"])
            for i, k in enumerate(sorted(keys)):
                w.writerow([f"C{i:06d}", all_first[k], k])
        with open(os.path.join(args.outdir, "mapping.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["old_key", "cluster_id"])
            for i, k in enumerate(sorted(keys)):
                w.writerow([k, f"C{i:06d}"])
        with open(os.path.join(args.outdir, "stats.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["total_keys", "clusters", "dedup_note"]) 
            w.writerow([len(keys), len(keys), "no_embeddings"])
        print("[MERGE] embeddings無しのため、単純結合で出力しました:", out_path)
        return

    # ベクトル行列
    vecs = np.stack([all_vecs[k] for k in emb_keys], axis=0).astype(np.float32)
    # 念のため正規化
    nrm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs = (vecs / nrm).astype(np.float32)

    clusters = cluster_by_threshold(emb_keys, vecs, float(args.alias_threshold))
    # cluster_id → メンバー(keys)
    cl_to_members: Dict[int, List[str]] = {}
    for k, c in zip(emb_keys, clusters):
        cl_to_members.setdefault(c, []).append(k)

    # embeddingsが無いキーは固有クラスタとして追加
    next_id = (max(cl_to_members.keys()) + 1) if cl_to_members else 0
    for k in keys:
        if k not in all_vecs:
            cl_to_members.setdefault(next_id, []).append(k)
            next_id += 1

    # cluster_idの並び替えと命名
    # 代表の最小時刻で昇順
    clusters_sorted: List[Tuple[int, dt.datetime]] = []
    for cid, members in cl_to_members.items():
        firsts = [parse_iso(all_first.get(m, "1970-01-01T00:00:00")) for m in members]
        tmin = min(firsts) if firsts else dt.datetime(1970, 1, 1)
        clusters_sorted.append((cid, tmin))
    clusters_sorted.sort(key=lambda x: x[1])

    cid_map: Dict[int, str] = {}
    for i, (old, _) in enumerate(clusters_sorted):
        cid_map[old] = f"C{i:06d}"

    # 出力: merged_first_seen.csv
    merged_path = os.path.join(args.outdir, "merged_first_seen.csv")
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "first_ts", "members"])  # membersはカンマ区切り
        for old, _ in clusters_sorted:
            members = cl_to_members[old]
            tmin = min([parse_iso(all_first.get(m, "1970-01-01T00:00:00")) for m in members])
            w.writerow([cid_map[old], tmin.isoformat(), ",".join(sorted(members))])

    # 出力: mapping.csv
    with open(os.path.join(args.outdir, "mapping.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_key", "cluster_id"])
        for old, _ in clusters_sorted:
            for m in sorted(cl_to_members[old]):
                w.writerow([m, cid_map[old]])

    # 出力: clusters.csv（冗長）
    with open(os.path.join(args.outdir, "clusters.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "size", "members"])
        for old, _ in clusters_sorted:
            mem = sorted(cl_to_members[old])
            w.writerow([cid_map[old], len(mem), ",".join(mem)])

    # 出力: stats.csv
    with open(os.path.join(args.outdir, "stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["total_keys", "clusters", "alias_threshold"]) 
        w.writerow([len(keys), len(clusters_sorted), float(args.alias_threshold)])

    print("[MERGE] saved:")
    print(" -", merged_path)
    print(" -", os.path.join(args.outdir, "mapping.csv"))
    print(" -", os.path.join(args.outdir, "clusters.csv"))
    print(" -", os.path.join(args.outdir, "stats.csv"))


if __name__ == "__main__":
    main()


