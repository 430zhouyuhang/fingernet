import os
import glob
import argparse
import numpy as np
from itertools import combinations

from match import TemplateMatcher, Minutia


def parse_args():
    parser = argparse.ArgumentParser(
        description="RANSAC仿射重投影误差评估（同指多次按压）"
    )
    parser.add_argument("--mnt_dir", type=str, required=True, help="细节点.mnt目录")
    parser.add_argument("--num_pairs", type=int, default=10, help="评估的样本对数量")
    parser.add_argument("--distance_tol", type=float, default=20.0, help="重投影距离阈值(像素)")
    parser.add_argument("--orientation_tol", type=float, default=25.0, help="方向阈值(度)")
    parser.add_argument("--max_pairs_per_finger", type=int, default=5, help="每个手指最多尝试的样本对数量")
    parser.add_argument("--match_threshold", type=float, default=0.65, help="模板匹配阈值")
    return parser.parse_args()


def find_mnt_files(mnt_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(mnt_dir, "*.mnt"))):
        name = os.path.splitext(os.path.basename(path))[0]
        parts = name.split("_")
        if len(parts) < 2:
            continue
        finger_id = parts[0]
        sample_id = "_".join(parts[1:])
        records.append((finger_id, sample_id, path))
    return records


def load_minutiae_map(matcher, records):
    cache = {}
    for finger_id, sample_id, path in records:
        minutiae = matcher.load_minutiae_from_mnt(path)
        if len(minutiae) == 0:
            continue
        cache[(finger_id, sample_id)] = minutiae
    return cache


def evaluate_reprojection(matcher, minutiae_a, minutiae_b, transform, dist_tol, ori_tol_rad):
    if transform is None:
        return 0, 0.0, 0.0
    pts_a = np.array([[m.x, m.y, 1.0] for m in minutiae_a])
    transformed = (pts_a @ transform.T)[:, :2]
    used_b = set()
    hit_errors = []
    for idx, (tx, ty) in enumerate(transformed):
        best_j = None
        best_d = None
        for j, mb in enumerate(minutiae_b):
            if j in used_b:
                continue
            dist = np.hypot(tx - mb.x, ty - mb.y)
            if dist > dist_tol:
                continue
            ori_diff = matcher._calculate_angle_difference(minutiae_a[idx].orientation, mb.orientation)
            if ori_diff > ori_tol_rad:
                continue
            if best_d is None or dist < best_d:
                best_d = dist
                best_j = j
        if best_j is not None:
            used_b.add(best_j)
            hit_errors.append(best_d)
    hit_count = len(hit_errors)
    denom = min(len(minutiae_a), len(minutiae_b))
    hit_rate = hit_count / denom
    mean_err = float(np.mean(hit_errors)) if hit_errors else 0.0
    return hit_count, hit_rate, mean_err


def pick_top_pairs(matcher, groups, minutiae_cache, max_pairs_per_finger, num_pairs):
    scored = []
    for finger_id, samples in groups.items():
        if len(samples) < 2:
            continue
        pairs = list(combinations(samples, 2))[:max_pairs_per_finger]
        for (sid_a, path_a), (sid_b, path_b) in pairs:
            key_a = (finger_id, sid_a)
            key_b = (finger_id, sid_b)
            if key_a not in minutiae_cache or key_b not in minutiae_cache:
                continue
            minutiae_a = minutiae_cache[key_a]
            minutiae_b = minutiae_cache[key_b]
            if len(minutiae_a) == 0 or len(minutiae_b) == 0:
                continue
            result = matcher.match_fingerprints(minutiae_a, minutiae_b)
            scored.append((result.score, finger_id, sid_a, sid_b, path_a, path_b, minutiae_a, minutiae_b, result))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:num_pairs]


def main():
    args = parse_args()
    matcher = TemplateMatcher(match_threshold=args.match_threshold)
    records = find_mnt_files(args.mnt_dir)
    if len(records) == 0:
        print("未找到.mnt文件")
        return
    groups = {}
    for finger_id, sample_id, path in records:
        groups.setdefault(finger_id, []).append((sample_id, path))
    minutiae_cache = load_minutiae_map(matcher, records)
    top_pairs = pick_top_pairs(
        matcher,
        groups,
        minutiae_cache,
        args.max_pairs_per_finger,
        args.num_pairs
    )
    if len(top_pairs) == 0:
        print("无可用样本对")
        return
    ori_tol_rad = np.radians(args.orientation_tol)
    print(f"选取分数最高的 {len(top_pairs)} 对样本进行重投影评估:")
    for rank, (score, fid, sid_a, sid_b, path_a, path_b, minutiae_a, minutiae_b, result) in enumerate(top_pairs, 1):
        hit, rate, mean_err = evaluate_reprojection(
            matcher,
            minutiae_a,
            minutiae_b,
            result.transform_matrix,
            args.distance_tol,
            ori_tol_rad
        )
        file_a = os.path.basename(path_a)
        file_b = os.path.basename(path_b)
        print(f"[{rank}] finger={fid} {sid_a}({file_a}) vs {sid_b}({file_b}) | 分数={score:.4f} | 内点数={result.num_matches} "
              f"| 命中={hit} | 命中率={rate*100:.2f}% | 平均误差={mean_err:.2f}px")


if __name__ == "__main__":
    main()

