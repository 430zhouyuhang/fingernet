<!-- f7c13269-806c-498f-b2ff-513fa7282f09 9b82f75a-b375-42cc-8125-ac842b0d884f -->
# 移动端部署优化方案

## 步骤

1. **裁剪部署入口**  

- 清理 `src/train_test_deploy.py` 中 `deploy()` 的可视化 / MAT 保存逻辑，仅保留生成 `.mnt` 的代码路径，确保输出专注于匹配所需特征。

2. **拆分并导出精简推理图**  

- 将 FingerNet 主干模型定义抽离成独立模块（例如 `src/model_export.py`），加载预训练权重后导出冻结图或 SavedModel，并转换为 TFLite / NNAPI 兼容格式（含量化、裁剪无用头）。

3. **端侧后处理优化**  

- 用纯 NumPy/ONNXRuntime 可移植实现重写 `label2mnt`、NMS 和方向重建逻辑，替换依赖 SciPy 的操作；引入批处理/向量化，减少内存占用。

4. **匹配器轻量化**  

- 在 `match/match.py` 中：
- 为 `TemplateMatcher` 增加可配置的 `k_neighbors`、模板数量、阈值以适配算力；
- 预编译局部模板并缓存到轻量二进制格式；
- 评估并实现 SIMD/并发友好版本的 RANSAC 与距离计算，或集成芯片厂商提供的加速库。