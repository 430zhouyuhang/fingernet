对目前 FVC2000 DB1_A 导出的细节点（单指约 40–90 个、图像分辨率 288×384、ridge 间距 ~10 像素）的经验，影响匹配结果最敏感的参数及建议范围如下：

- `match_threshold`：直接决定判定成败，0.55–0.75 常用；细节点较杂或低质量图建议 0.55–0.65，高质量或想压 FAR 可到 0.7 以上。
- `k_neighbors`：控制局部模板信息量。5–8 比较稳；细节点少时可以降到 3–4，细节点密集时可升到 10。
- `template_radius`：相当于局部结构尺度；FVC2000 DB1_A 每像素 500dpi，80–150 像素较合适，太小模板不完整，太大易混入噪声。
- `distance_tolerance` / `angle_tolerance` / `orientation_tolerance`：直接影响局部匹配。距离容差建议 10–20 像素，角度/方向容差 25°–40°。图像配准精度越高可适当收紧。
- `max_distance`：RANSAC 内点阈值，对几何验证非常关键。30–60 像素常用；细节点定位误差大时放宽，反之收紧以降低假匹配。
- `ransac_runs`：影响稳定性与耗时。2–5 足够；数据噪声大或要高可靠性时可提高到 8–10。
- `templates_per_finger`： enrollment 样本数。1–3 常见；指纹采集波动大时建议 2–4 以提升召回。
- `estimated_inlier_ratio`：影响 RANSAC 迭代数。0.2–0.4 对 FVC2000 DB1_A 合理；越低迭代越多但更稳。
- `min_inliers_for_early_stop_ratio` 与 `min_inliers_for_early_stop_min`：提前停止阈值。比例 0.25–0.4、最少内点 4–8；细节点少时适当降低。
- `min_iterations_before_stop_ratio` / `min_iterations_before_stop_min`：迭代保底量。0.2–0.4、最少 30–60；分数波动大时略增。
- `early_stop_inlier_ratio`、`required_consecutive`：避免误停。内点比例 0.6–0.8、连续次数 2–4；值越高越保守。
- `min_inliers_for_result`、`min_inlier_ratio_for_result`：最终接受门槛。内点数 4–8、比例 0.25–0.4；FRR 高时可放宽。
- `scale_min` / `scale_max`：仿射缩放限制，FVC 数据采集精度高，0.5–2.0 足够；若担心严重畸变可保持现状 0.3–3.0。
- `spread_min_radius`：内点空间分布阈值，控制是否集中在小区域。10–20 像素比较合适，低于 ridge 间距两倍会过严。

上述范围均基于现有输出的细节点密度和图像尺度，可在实际评估中围绕这些区间微调以平衡 FAR/FRR。


调参方向：
- 提升判决阈值：`match_threshold` 提到 0.78–0.80，可先跑 0.78 观察；若 FAR 仍高，逐步加到 0.82。
- 收紧局部匹配：将 `distance_tolerance` 调到 12，`angle_tolerance` 25，`orientation_tolerance` 25；必要时 `k_neighbors` 维持 5 但可尝试 6 以增加判定信息。
- 几何验证更严格：`max_distance` 设 32；`min_inliers_for_result` 至少 7，`min_inlier_ratio_for_result` 设 0.35；同时把 `min_inliers_for_early_stop_ratio` 提到 0.4，`early_stop_inlier_ratio` 0.8，抑制偶然对齐。
- 模板多样性：若 enrollment 数据允许，可把 `templates_per_finger` 设 2（或从每指选两个质量较好的样本）以抵消更严规则导致的 FRR 上升。
- 其他保持默认即可（`estimated_inlier_ratio` 0.3），如需要额外稳定性可把 `ransac_runs` 提到 5。

这些改动会明显压低 FAR；如果 FRR 逼近 5%，优先微调 `match_threshold`（如降到 0.76）或略放宽 `max_distance`（至 34）。