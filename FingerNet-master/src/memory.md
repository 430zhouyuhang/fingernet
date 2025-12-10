1. **模型权重与 TF 计算图（≈ 800–1000 MB）**  
   - FingerNet 主网参数、BatchNorm 动态变量、卷积缓存，以及 TF1 图本身的结构元数据，在 `predict()` 第一次运行时整体加载到内存里，占比最大。

2. **推理中间特征图（张量）与临时 workspace（≈ 15–30 MB）**  
   - `enhance_img`、`ori_out_1/2`、`seg_out`、`mnt_*` 等 float32 张量，以及 TF 为卷积/反卷积准备的临时缓冲区（尤其在 CPU 上），总体十几到几十 MB。

3. **Python 侧的 NumPy 拷贝与后处理缓存（≈ 200–400 MB）**  
   - `np.squeeze`、`label2mnt`、NMS、方向图转换、`ori_highest_peak_numpy` 等操作都会对张量做副本；
   - `mnt_writer`、`draw_ori_on_img`、`draw_minutiae`、`imageio.imwrite`、`io.savemat` 在写磁盘前保留整张图，这些 8bit/float 数组叠加后迅速达到数百 MB。

4. **日志与统计结构（≈ 50–100 MB）**  
   - `time_stats`、`memory_stats`、路径/字符串、Matplotlib 绘图缓存、`runtime_memory_stats` 等 Python 对象长期驻留，虽然单个很小，但成千上百条记录也会占几十 MB。

5. **解释器与依赖库基础占用（≈ 500–600 MB）**  
   - Python 解释器、NumPy/SciPy/OpenCV 动态库常驻内存；  
   - Windows 进程的工作集管理、JIT/BLAS 缓冲也会在 300–400 MB 以上。

20 MB 的 .model 文件只是“参数权重”的存档体积，它是经过序列化、压缩、只包含数值本身的。加载运行后，内存里同时存在： • 解压后的参数矩阵：float32/float64 形式展开，体积本身就比序列化文件大；再加上 Keras/TF 为每层创建的 Variable、Tensor 对象、依赖指针。  • BatchNorm/优化器的运行缓冲：均值/方差、动量、slot 变量等都会复制一份，与权重一样大，甚至多几倍。  • 计算图与执行计划：TF1 会把每个节点、边、属性都存储为 C++ 结构体，再附带 kernel 注册表和 shape 推断结果，占用远大于权重文件。  • 工作区与缓存：卷积/反卷积需要的 im2col、FFT、临时 tensor，第一次 predict() 时根据输入尺寸分配，并持续保留以便复用。   所以虽然磁盘上只有 20 MB，运行过程中模型相关部分会扩张到几百 MB，配合其他中间结果就逼近 1 GB。

   统计某一张图进行匹配之前和匹配完成之后的增加的内存峰值。


   大约 500 MB 的匹配“运行内存增量”主要来自以下几块（估算为典型 FVC 数据集、默认参数配置下的数量级，仅作参考）：

- **模板/细节点缓存（≈ 200–250 MB）**  
  `match/test_fingerprint_matching.py` 会把同一个手指的多组 minutiae 读入内存，`TemplateMatcher.compute_templates()` 为每个细节点生成 `np.ndarray`（距离、角度、方向差等），并将所有模板一起序列化到 `multi_templates`。当指纹数量多、`k_neighbors=5` 时，这部分就会占据几百 MB 中的最大份额。

- **查询匹配工作区（≈ 80–120 MB）**  
  在 `match_with_stored_template()` 内部，需要把查询指纹的 minutiae、局部模板、候选匹配对以及 RANSAC 相关矩阵全部保存在 numpy 数组中，便于多次迭代使用。尤其 `find_template_matches()` 与 `geometric_verification()` 做大量矩阵运算，会分配数十 MB 的临时数组。

- **多指纹队列及统计缓存（≈ 50–80 MB）**  
  `sample_details`、`genuine_scores`、`impostor_scores` 会持续 append，保存每次匹配的得分、匹配对、索引等结构化数据。虽然单条记录不大，但几千条记录后就能达到几十 MB。

- **图像与可视化缓存（≈ 40–60 MB）**  
  `cv2.imread` 加载的原始指纹图像（灰度 640×640）每张约 0.4 MB，但在可视化阶段会用 matplotlib 将多个子图渲染到内存（默认使用 float64 缓冲），并在保存前保留整张 2×1 拼图，因此会产生十几到几十 MB 的临时占用。

- **解释器 / 依赖库基线（≈ 60–80 MB 增量）**  
  第一次调用匹配或绘图时，NumPy/SciPy/Matplotlib/OpenCV 等库会创建额外缓存与句柄；虽然这些库早已加载，但即刻运行的函数还会多申请一些工作集，从而体现在“增量”里。

综上，加总就接近 500 MB。若希望继续压缩，可以：限制一次加载的手指数量、在完成一批匹配后显式 `del` 模板/细节点并 `gc.collect()`、关闭可视化或改为轻量绘图、减少 `sample_details` 中保存的字段等。