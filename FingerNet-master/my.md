下面是对该 README 的详细总结与提炼，方便你快速理解与上手使用 FingerNet：

# 项目概览

- **名称**：FingerNet —— 一个通用的深度卷积网络，用于提取指纹表征
- **可输出**：方向场（orientation field）、前景分割（segmentation）、增强指纹（enhenced fingerprint）、细节点/特征点（minutiae）
- **适用场景**：平面/滚动指纹与潜指纹（rolled/slap & latent）
- **开源协议**：MIT
- **引用**：Tang et al., *IJCB 2017* 的会议论文 BibTeX 已给出

# 主要结果（示例）

| 训练集    | 测试集   | Precision | Recall | 卷积耗时/图 | 后处理耗时/图 |
| --------- | -------- | --------- | ------ | ----------- | ------------- |
| CISL24218 | FVC2004  | 76%       | 80%    | 674 ms      | 285 ms        |
| CISL24218 | NISTSD27 | 63%       | 63%    | 183 ms      | 885 ms        |


 NISTSD27目前无法获取

**说明**：

- 批大小 > 1 可加速卷积阶段；
- 后处理包含方向选择、分割膨胀、细节点 NMS；
- CISL24218 为内部数据集，未公开。

# 运行环境与依赖

## 原始要求（旧版）

- **Python 2.7**，依赖：cv2、numpy、scipy、matplotlib、pydot、graphviz
- **TensorFlow 1.0.1**
- **Keras 2.0.2**

## TF2 兼容版（新增在 `src/`）

- 使用 `tf.compat.v1` + `tensorflow.keras`，保持网络结构与 I/O 不变。

- 推荐（CPU）环境：

  ```bash
  conda create -n fingernet-tf2 python=3.10 -y
  conda activate fingernet-tf2
  pip install "tensorflow<2.16" numpy scipy matplotlib imageio opencv-python pydot graphviz
  ```

- 重要变更：

  - 图像读写改为 `imageio`（替代 `scipy.misc.imread/imsave`）
  - 采用 Keras legacy 优化器：`tf.keras.optimizers.legacy.Adam/SGD`
  - 高斯窗口用 `scipy.signal.windows.gaussian`
  - **预训练权重**：`models/released_version/Model.model` 若非 TF Checkpoint/h5，不能直接 `load_weights`；可先设 `pretrain=None` 跑通，或准备匹配格式的权重。

# 硬件需求

- GPU：Titan/Titan Black/Titan X/K20/K40/K80（例示）
- 显存参考：FVC2002DB2A ≈ 2GB；NISTSD27 ≈ 5GB

# 快速上手（推理 Demo / 测试 / 训练）

> 工作目录建议切到 `src/`，以确保相对路径（如 `../datasets/`）正确。

## 推理 Demo（部署）

```bash
cd src
python train_test_deploy.py 0 deploy
```

- `0` 为 GPU ID，可改为其他编号；
- 代码会对 `datasets/` 中自带样例运行，并打印三段时间：加载+卷积、后处理（方向选择+分割后处理+NMS）、可视化绘制（draw）。
- 想测试其他数据集：修改 `train_test_deploy.py` 中的 `deploy_set=[...]` 所指的数据集文件夹。

## 测试（需要分割/细节点标签）

```bash
python train_test_deploy.py 0 test
```

- 与部署不同，测试需要至少 **细节点（mnt）** 和 **分割** 标签；
- 通过修改 `test_set=[...]` 指定测试数据集。

## 训练（需要完整标签）

```bash
python train_test_deploy.py 0 train
```

- 最多 100 个 epoch，收敛将提前停止（early stop）。

# 数据准备与目录结构

- 输入与目录结构
  - 训练集根目录：`datasets/<dataset-name>/`
  - 必需子目录（同名对齐，文件基名一致）：
    - `images/`：指纹灰度图，格式 `.bmp`，例如 `A001.bmp`
    - `seg_labels/`：分割标签，格式 `.png`，与图像同名同尺寸（前景=255，背景=0），例如 `A001.png`
    - `ori_labels/`：方向标签图，格式 `.bmp`，与图像同名同尺寸（与原图对齐），例如 `A001.bmp`
    - `mnt_labels/`：细节点标签，格式 `.mnt`，与图像同名（文本文件），例如 `A001.mnt`

- 文件命名要求
  - 四个目录中的文件“基名”必须完全一致（如 `A001.bmp`、`A001.png`、`A001.bmp`、`A001.mnt`）。
  - 扩展名固定：图像 `.bmp`，分割 `.png`，方向 `.bmp`，细节点 `.mnt`。

- .mnt 文件格式（逐行）
  - 第1行：图像名（可被忽略，代码不依赖）
  - 第2行：头信息（可被忽略，代码不依赖）
  - 之后每行一个细节点：`w h o`
    - `w`、`h` 为像素坐标（列、行），代码会取整
    - `o` 为方向，单位“弧度”（建议范围 [0, 2π)）
  - 说明：读取时跳过前两行，仅使用后续的 `w h o` 行

- 数据内容与尺寸要求
  - 图像为单通道灰度图；代码会归一化到 [0, 1]
  - 分割标签值应为 0/255（代码会阈值化为 0/1）
  - 方向标签图用于传统方向估计（与原图对齐、同尺寸）
  - 细节点坐标需位于图内，且非边界（代码训练时会过滤边缘8像素）
  - 尺寸需为 8 的倍数；若不是，代码会在加载阶段自动“填充/对齐”为最近的 8 倍尺寸（不必手动裁剪）

- 多数据集训练
  - 在 `train_test_deploy.py` 顶部设置 `train_set = ['../datasets/<dataset>/', ...]`
  - 可通过 `train_sample_rate` 控制各数据集采样比例（None=等权）

- 运行入口（示例）
  - 工作目录建议 `src/`
  - 命令：`python train_test_deploy.py 0 train`（参数：GPU_ID=0，模式=train）



# 实用提示与常见坑

- **批处理**：推理阶段适当增大 batch size 可显著降低卷积时间（受显存限制）。
- **权重格式**：如直接加载历史权重失败，使用 `pretrain=None` 先跑通流程，再按 TF2 期望格式准备权重。
- **路径问题**：务必从 `src/` 作为工作目录运行，避免相对路径失效。
- **可视化耗时**：`draw` 阶段也计时，若仅关心数值输出，可在代码中按需裁减绘图步骤。
- **IDE 运行**：PyCharm 可在 Run/Debug 配置中设置参数：`0 deploy` / `0 test` / `0 train`。

------









deploy 输入 输出

- 输出目录
  - 路径：`output/<时间戳>/<set_name>/`
  
  - 其中 `<set_name>` 对应 `deploy_set` 的序号（0,1,2, …）。

  - deploy_set 是train_test_deploy.py第52-53行定义的一个列表：
    0/ - 对应 deploy_set[0] (NISTSD27/images/)  
  
     1/ - 对应 deploy_set[1] (CISL24218/)  
  
     2/ - 对应 deploy_set[2] (FVC2002DB2A/)  
  
     3/ - 对应 deploy_set[3] (NIST4/)


- 图像文件
  - `<name>_enh.png`：增强图（相位增强×分割掩码）
    - 单通道8位，0–255。高亮处为纹线方向能量较强的区域。
  - `<name>_seg.png`：分割结果（前景/背景）
    - 单通道8位，0 或 255。指纹前景区域为255。
  - `<name>_ori.png`：方向场可视化
    - 在原图上以箭头显示局部主方向（步长16）。只在前景位置绘制。

- 结构化结果
  - `<name>.mnt`：检测到的细节点列表（文本）
    - 第1行：图像名
    - 第2行：N H W（N=细节点数，H/W=图像高宽）
    - 第3..N+2行：`x y o`
      - `x`、`y`：像素坐标（列、行）
      - `o`：方向（弧度，约 [0, 2π)）
    - 用途：后续匹配、评估或入库。
  - `<name>.mat`：MATLAB 格式（键值）
    - `orientation`：最终方向角度矩阵（弧度，尺寸≈原图/步长上采样到原图）
    - `orientation_distribution_map`：方向分布（网络 ori_out_1 的概率图，形状 ≈ [H/8, W/8, 90]）

- 日志
  - `output/<时间戳>/log.log` 与控制台：
    - `Predicting <set_name>:`：开始处理某个集合
    - 每张图的处理计时：`load+conv`（读取+前向）、`seg-postpro+nms`（后处理）、`draw`（可视化保存）

- 数值/尺寸说明
  - 网络内部特征以 8 为步长；`seg_out` 在保存前上采样到原图。
  - 方向分布通道数：
    - 方向（ori_out）：90通道，对应 1°,3°,5°,…,179°（步长2°，表示 0–180° 的等效方向）。
  - 细节点输出：
    - `mnt_s_out` 为置信图（阈值0.5后做NMS），`mnt_w_out/mnt_h_out` 给出 8×8 cell 内的子像素偏移（0–7），`mnt_o_out` 给出 180 通道方向（0–360°，步长2°）。

- 常见核对方式
  - 分割质量：查看 `<name>_seg.png` 是否准确覆盖指纹区。
  - 方向场：查看 `<name>_ori.png` 箭头是否与纹线走向一致。
  - 细节点：对照 `<name>_mnt.png`（若在测试流程中，会同时有 `_mnt_gt.png` 作为对比）。





train

- 数据目录与命名
  - 训练集根目录：`datasets/<dataset-name>/` 该项目使用CISL24218，不可公开获取
  - 必备子目录与文件格式（四者“同名对齐”）：
    - `images/*.bmp`：指纹灰度图（单通道）
    - `seg_labels/*.png`：分割标签（与图同名同尺，前景=255，背景=0）
    - `ori_labels/*.bmp`：方向标签图（与图同名同尺，用于传统方向估计的输入对齐）
    - `mnt_labels/*.mnt`：细节点标签（与图同名的文本）
  - .mnt 文件内容（文本）：
    - 第1行：图像名（例如 `A001`）
    - 第2行：`N H W`（细节点数N、图像高H、图像宽W）
    - 第3..N+2行：`x y o`
      - `x y` 为像素坐标（列、行，代码内会取整）
      - `o` 为方向（弧度，建议范围 [0, 2π)）

- 尺寸与像素要求
  - 原始图为单通道灰度；训练时按 `img/255.0` 归一化到 [0,1]
  - 分割标签值用 0/255，代码会转为 0/1
  - 输入尺寸建议为 8 的倍数；若不是，代码会在加载时“填充到最近的 8 的倍数”（用图像均值/0 填充）
  - 细节点位于图内且远离边界 ≥8 像素（代码会过滤边缘）

- 数据增强（默认开启 aug=0.7，训练时）
  - 随机旋转：0–360°
  - 随机平移：每个轴最多图像高/宽的 1/4
  - 不满足目标尺寸时执行随机或居中填充对齐（到 8 的倍数）

- 批数据张量形状（以单批 B=batch_size，H,W 为对齐后的尺寸）
  - 模型输入
    - `image`: (B, H, W, 1)，float32，范围 [0,1]
  - 中间监督（方向粗标签由传统模块 `tra_ori_model.predict(alignment)` 生成）
    - `alignment`: (B, H, W, 1)，来自 `ori_labels`（无则为全0），作为方向估计的输入
  - 训练标签（在 `load_data` 内构造，特征图步幅=8）
    - `label_seg`: (B, H/8, W/8, 1) — 分割 0/1（由 `seg_labels` 下采样 8×8）
    - `label_ori`: (B, H/8, W/8, 90) — 方向分布（0–180°，步长2°，以高斯窗生成，且仅在前景+有对齐处有效）
    - `label_ori_o`: (B, H/8, W/8, 90) — 细节点方向的“等效方向”分布（0–180°）
    - `label_mnt_o`: (B, H/8, W/8, 180) — 细节点真正方向分布（0–360°，步长2°）
    - `label_mnt_w`: (B, H/8, W/8, 8) — 细节点 x 偏移（0–7，对应 8×8 cell 内位置）
    - `label_mnt_h`: (B, H/8, W/8, 8) — 细节点 y 偏移（0–7）
    - `label_mnt_s`: (B, H/8, W/8, 1) — 细节点置信度标签，取值 {-1,0,1}
      - 初始为 {0/1}（是否存在细节点），再将 0 置为 -1，且对邻域做一次最大滤波并平均，使边缘无关区域变 0（no care）
      - 损失里会把 -1 当作 0 处理（忽略负样本损失）

- 损失与指标（核心）
  - `ori_loss`：方向分布的加权交叉熵 + 一致性（邻域方向一致性）
  - `seg_loss`：分割加权交叉熵 + 平滑（拉普拉斯核 L1）
  - `mnt_s_loss`：细节点置信度加权交叉熵（-1/0/1 标签，-1 将在损失中置为 0）
  - 度量：
    - 方向/细节点角度 Top-1 精度（阈值 10°/20°）
    - 分割精度（正/负/总体）
    - 偏移平均误差（w/h 的 argmax 与 GT 差）

- 训练过程产生的输出（写到 `output/<时间戳>/`）
  - `log.log`：loss 与 metrics 的持续记录
  - `model.png`：网络结构图
  - 周期性保存权重（`save_weights`，文件名为步号）
  - 每处理若干步会触发快速测试（按 `test()`），写出测试集的指标日志与可视化结果（同 test 模式）

- 关键一致性约束（易错点）
  - 四类文件必须同名对齐（例如 `A001.bmp`、`A001.png`、`A001.bmp`、`A001.mnt`）
  - `.mnt` 的方向单位为弧度，训练中会转换到度并映射到高斯分布标签
  - 目录相对路径依赖工作目录为 `src/`（即从 `src/` 运行），否则 `../datasets/...` 会解析失败
  - 若没有可用预训练权重，训练从随机初始化开始；可将 `pretrain` 设为权重前缀以 `by_name=True` 加载兼容层参数

- 小示例（.mnt）
  - 第1行：`A001`
  - 第2行：`N H W` 例如 `42 512 512`
  - 第3..：`x y o` 例如 `123 245 1.570796`（x=123,y=245,o=π/2）





test

- 输入（datasets/<name>/，同名对齐）
  - images/*.bmp：灰度指纹图（单通道）
  - seg_labels/*.png：分割标签（0/255，与图同名同尺寸）
  - ori_labels/*.bmp：方向标签图（与图同名同尺寸，用于传统方向估计的输入对齐）
  - mnt_labels/*.mnt：细节点标签（文本，行3..为 x y o，o 为弧度）
  - 预训练权重（脚本中 pretrain 指定，若不可用也能跑但指标无意义）

- 预测输出（写入 output/<时间戳>/）
  - 可视化/图片（单图）
    - <name>_ori.png：方向场可视化（原图上以箭头显示，步长16，仅前景）
    - <name>_mnt.png：预测细节点（红框+方向线）
    - <name>_mnt_gt.png：GT 细节点可视化（对照用，若 draw=True）
  - 控制台与日志里会打印每张图的损失与指标（见下）

- 内部张量（模型predict的主要输出，供理解）
  - ori_out_1/ori_out_2：(H/8,W/8,90)，方向分布[0,1]，表示 0–180°（步长2°）
  - seg_out：(H/8,W/8,1)，分割概率[0,1]（保存前会上采样）
  - mnt_o_out：(H/8,W/8,180)，细节点方向分布（0–360°，步长2°）
  - mnt_w_out/mnt_h_out：(H/8,W/8,8)，x/y 偏移（0–7，对应8×8 cell内位置）
  - mnt_s_out：(H/8,W/8,1)，细节点置信度

- 控制台输出含义（典型一张图）
  - loss: 总损失
  - 各分支 loss：
    - ori_out_1_loss：方向分布（含一致性）
    - ori_out_2_loss：方向（辅助头）交叉熵
    - seg_out_loss：分割加权CE+平滑
    - mnt_o_out_loss：细节点方向分布交叉熵
    - mnt_w_out_loss / mnt_h_out_loss：偏移分布交叉熵
    - mnt_s_out_loss：细节点置信度加权CE
  - 指标（metrics）：
    - ori_out_1_ori_acc_delta_10 / _20：方向Top-1落在±10°/±20°的比例
    - seg_out_seg_acc_pos/neg/all：分割正/负/总体像素准确率
    - mnt_o_out_mnt_acc_delta_10 / _20：细节点方向准确率（±10°/±20°）
    - mnt_w_out_mnt_mean_delta、mnt_h_out_mnt_mean_delta：偏移平均误差（cell内 0–7 的差值）
  - After_nms（细节点NMS后对GT评估）：
    - precision / recall / f1-measure：基于距离与方向阈值匹配得到
    - location_dis：匹配样本的平均位置误差（像素）
    - orientation_delta：匹配样本的平均方向误差（弧度）
  - Average testing results：对前面若干张图的上述 loss/metrics 与 After_nms 汇总平均

- 常见核对
  - 若 pretrain 加载失败（DATA_LOSS 提示），结果来自随机初始化，指标仅用于流程验证。
  - 可通过 draw=True 生成可视化（ori/mnt/mnt_gt）以肉眼核对预测质量。

