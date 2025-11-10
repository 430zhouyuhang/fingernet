"""
使用db4_b数据集测试模板匹配算法
计算FAR和FRR指标
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加match模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from match import TemplateMatcher, Minutia

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_mnt_files(dataset_dir: str, output_dir: str) -> List[Tuple[str, str, str]]:
    """
    查找所有图像和对应的mnt文件
    
    Args:
        dataset_dir: 数据集目录（包含.bmp文件）
        output_dir: 输出目录（包含.mnt文件）
        
    Returns:
        [(finger_id, sample_id, image_path, mnt_path), ...]
    """
    files = []
    
    # 获取所有bmp文件
    bmp_files = [f for f in os.listdir(dataset_dir) if f.endswith('.bmp')]
    
    for bmp_file in sorted(bmp_files):
        # 解析文件名：00000_00.bmp -> finger_id=00000, sample_id=00
        name_parts = bmp_file.replace('.bmp', '').split('_')
        if len(name_parts) == 2:
            finger_id = name_parts[0]
            sample_id = name_parts[1]
            
            image_path = os.path.join(dataset_dir, bmp_file)
            mnt_path = os.path.join(output_dir, bmp_file.replace('.bmp', '.mnt'))
            
            # 检查mnt文件是否存在
            if os.path.exists(mnt_path):
                files.append((finger_id, sample_id, image_path, mnt_path))
            else:
                print(f"警告: 未找到mnt文件: {mnt_path}")
    
    return files


def build_templates_and_save(matcher: TemplateMatcher, 
                            files: List[Tuple[str, str, str, str]],
                            template_dir: str,
                            use_multi_template: bool = True):
    """
    为所有指纹构建模板并保存（支持多模板策略）
    
    Args:
        matcher: 模板匹配器
        files: 文件列表
        template_dir: 模板保存目录
        use_multi_template: 是否使用多模板（同一手指的多个样本合并为一个模板）
    """
    os.makedirs(template_dir, exist_ok=True)
    
    print("="*60)
    if use_multi_template:
        print("构建并保存多模板（每个手指包含多个样本）...")
    else:
        print("构建并保存模板（每个样本单独保存）...")
    print("="*60)
    
    # 按手指ID分组
    finger_groups = {}
    for finger_id, sample_id, image_path, mnt_path in files:
        if finger_id not in finger_groups:
            finger_groups[finger_id] = []
        finger_groups[finger_id].append((sample_id, image_path, mnt_path))
    
    if use_multi_template:
        # 多模板策略：为每个手指构建包含所有样本的多模板
        for finger_id in sorted(finger_groups.keys()):
            samples = finger_groups[finger_id]
            multi_templates = []
            total_minutiae = 0
            
            for sample_id, image_path, mnt_path in samples:
                # 加载细节点
                minutiae = matcher.load_minutiae_from_mnt(mnt_path)
                
                if len(minutiae) == 0:
                    print(f"跳过 {finger_id}_{sample_id}: 无细节点")
                    continue
                
                # 构建模板
                templates = matcher.compute_templates(minutiae)
                multi_templates.append((templates, minutiae))
                total_minutiae += len(minutiae)
            
            if len(multi_templates) == 0:
                print(f"跳过 {finger_id}: 无有效样本")
                continue
            
            # 保存多模板（包含样本ID，用于避免数据泄露）
            template_file = os.path.join(template_dir, f"{finger_id}_multi.pkl")
            sample_ids = [f"{finger_id}_{sample_id}" for sample_id, _, _ in samples]  # 保存样本ID
            metadata = {
                'finger_id': finger_id,
                'num_samples': len(multi_templates),
                'total_minutiae': total_minutiae
            }
            matcher.save_multi_template(template_file, multi_templates, metadata, sample_ids)
            
            print(f"已保存多模板: {finger_id} (包含 {len(multi_templates)} 个样本, 共 {total_minutiae} 个细节点)")
    else:
        # 单模板策略：每个样本单独保存（向后兼容）
        for finger_id, sample_id, image_path, mnt_path in files:
            # 加载细节点
            minutiae = matcher.load_minutiae_from_mnt(mnt_path)
            
            if len(minutiae) == 0:
                print(f"跳过 {finger_id}_{sample_id}: 无细节点")
                continue
            
            # 构建模板
            templates = matcher.compute_templates(minutiae)
            
            # 保存模板
            template_file = os.path.join(template_dir, f"{finger_id}_{sample_id}.pkl")
            metadata = {
                'finger_id': finger_id,
                'sample_id': sample_id,
                'num_minutiae': len(minutiae)
            }
            matcher.save_template(templates, minutiae, template_file, metadata)
            
            print(f"已保存: {finger_id}_{sample_id} ({len(minutiae)}个细节点)")


def compute_far_frr(matcher: TemplateMatcher,
                    files: List[Tuple[str, str, str, str]],
                    template_dir: str,
                    threshold: float = 0.5) -> Tuple[float, float, List[float], List[float]]:
    """
    计算FAR和FRR
    
    Args:
        matcher: 模板匹配器
        files: 文件列表
        template_dir: 模板目录
        threshold: 匹配阈值
        
    Returns:
        (FAR, FRR, genuine_scores, impostor_scores)
    """
    print("="*60)
    print("计算FAR和FRR...")
    print("="*60)
    
    genuine_scores = []  # 正样本分数（同一手指不同样本）
    impostor_scores = []  # 负样本分数（不同手指）
    
    # 按手指ID分组
    finger_groups = {}
    for finger_id, sample_id, image_path, mnt_path in files:
        if finger_id not in finger_groups:
            finger_groups[finger_id] = []
        finger_groups[finger_id].append((sample_id, image_path, mnt_path))
    
    finger_ids = sorted(finger_groups.keys())
    num_fingers = len(finger_ids)
    
    print(f"找到 {num_fingers} 个手指，每个手指有多个样本")
    
    # 计算正样本分数（Genuine）
    # 正样本：同一手指的所有样本对之间的匹配
    print("\n计算正样本分数（同一手指所有样本对之间的匹配）...")
    genuine_count = 0
    
    # 检查是否使用多模板
    use_multi_template = False
    for finger_id in finger_ids[:1]:  # 检查第一个手指
        multi_template_file = os.path.join(template_dir, f"{finger_id}_multi.pkl")
        if os.path.exists(multi_template_file):
            use_multi_template = True
            break
    
    if use_multi_template:
        # 多模板策略：使用多模板进行匹配（留一法，避免数据泄露）
        print("使用多模板策略进行匹配（留一法）...")
        for finger_id in finger_ids:
            samples = finger_groups[finger_id]
            if len(samples) < 2:
                continue
            
            # 加载多模板
            multi_template_file = os.path.join(template_dir, f"{finger_id}_multi.pkl")
            if not os.path.exists(multi_template_file):
                continue
            
            # 测试所有样本作为查询指纹（留一法：排除查询样本本身）
            for sample_id, _, mnt_path in samples:
                query_minutiae = matcher.load_minutiae_from_mnt(mnt_path)
                if len(query_minutiae) == 0:
                    continue
                
                # 与多模板匹配，排除查询样本本身（避免自己匹配自己）
                query_sample_id = f"{finger_id}_{sample_id}"  # 构建查询样本ID
                result = matcher.match_with_stored_template(
                    multi_template_file, 
                    query_minutiae,
                    exclude_query_sample_id=query_sample_id  # 排除查询样本ID，避免数据泄露
                )
                genuine_scores.append(result.score)
                genuine_count += 1
                
                if genuine_count % 50 == 0:
                    print(f"  已处理 {genuine_count} 个正样本匹配...")
    else:
        # 单模板策略：测试所有样本对
        for finger_id in finger_ids:
            samples = finger_groups[finger_id]
            if len(samples) < 2:
                continue
            
            # 测试所有样本对（避免重复：i < j）
            for i in range(len(samples)):
                sample_id1, _, mnt_path1 = samples[i]
                template_file = os.path.join(template_dir, f"{finger_id}_{sample_id1}.pkl")
                
                if not os.path.exists(template_file):
                    continue
                
                for j in range(i + 1, len(samples)):
                    sample_id2, _, mnt_path2 = samples[j]
                    query_minutiae = matcher.load_minutiae_from_mnt(mnt_path2)
                    if len(query_minutiae) == 0:
                        continue
                    
                    result = matcher.match_with_stored_template(template_file, query_minutiae)
                    genuine_scores.append(result.score)
                    genuine_count += 1
                    
                    if genuine_count % 50 == 0:
                        print(f"  已处理 {genuine_count} 个正样本匹配...")
    
    print(f"正样本匹配总数: {len(genuine_scores)}")
    
    # 计算负样本分数（Impostor）
    # 负样本：不同手指的所有样本对之间的匹配
    print("\n计算负样本分数（不同手指所有样本对之间的匹配）...")
    impostor_count = 0
    # 限制负样本数量，避免计算时间过长
    # 对于10个手指，每个8个样本，全部组合是 10*9/2 * 8*8 = 2880对
    max_impostor = min(2000, len(genuine_scores) * 5)  # 限制负样本数量
    
    # 检查是否使用多模板
    use_multi_template = False
    for finger_id in finger_ids[:1]:  # 检查第一个手指
        multi_template_file = os.path.join(template_dir, f"{finger_id}_multi.pkl")
        if os.path.exists(multi_template_file):
            use_multi_template = True
            break
    
    if use_multi_template:
        # 多模板策略：使用多模板进行匹配（负样本不需要排除，因为不同手指）
        for i, finger_id1 in enumerate(finger_ids):
            samples1 = finger_groups[finger_id1]
            if len(samples1) == 0:
                continue
            
            # 加载多模板
            multi_template_file1 = os.path.join(template_dir, f"{finger_id1}_multi.pkl")
            if not os.path.exists(multi_template_file1):
                continue
            
            # 与其他手指匹配
            for finger_id2 in finger_ids[i+1:]:
                samples2 = finger_groups[finger_id2]
                if len(samples2) == 0:
                    continue
                
                # 测试所有样本作为查询指纹（不同手指，不需要排除）
                for sample_id2, _, mnt_path2 in samples2:
                    query_minutiae = matcher.load_minutiae_from_mnt(mnt_path2)
                    if len(query_minutiae) == 0:
                        continue
                    
                    # 与多模板匹配（不同手指，不需要排除）
                    result = matcher.match_with_stored_template(
                        multi_template_file1, 
                        query_minutiae,
                        exclude_query_sample_id=None  # 不同手指，不需要排除
                    )
                    impostor_scores.append(result.score)
                    impostor_count += 1
                    
                    if impostor_count >= max_impostor:
                        break
                    
                    if impostor_count % 100 == 0:
                        print(f"  已处理 {impostor_count} 个负样本匹配...")
                
                if impostor_count >= max_impostor:
                    break
            
            if impostor_count >= max_impostor:
                break
    else:
        # 单模板策略：测试所有样本对组合
        for i, finger_id1 in enumerate(finger_ids):
            samples1 = finger_groups[finger_id1]
            if len(samples1) == 0:
                continue
            
            # 与其他手指匹配
            for finger_id2 in finger_ids[i+1:]:
                samples2 = finger_groups[finger_id2]
                if len(samples2) == 0:
                    continue
                
                # 测试所有样本对组合
                for sample_id1, _, mnt_path1 in samples1:
                    template_file = os.path.join(template_dir, f"{finger_id1}_{sample_id1}.pkl")
                    
                    if not os.path.exists(template_file):
                        continue
                    
                    for sample_id2, _, mnt_path2 in samples2:
                        query_minutiae = matcher.load_minutiae_from_mnt(mnt_path2)
                        if len(query_minutiae) == 0:
                            continue
                        
                        result = matcher.match_with_stored_template(template_file, query_minutiae)
                        impostor_scores.append(result.score)
                        impostor_count += 1
                        
                        if impostor_count >= max_impostor:
                            break
                        
                        if impostor_count % 100 == 0:
                            print(f"  已处理 {impostor_count} 个负样本匹配...")
                    
                    if impostor_count >= max_impostor:
                        break
                
                if impostor_count >= max_impostor:
                    break
            
            if impostor_count >= max_impostor:
                break
    
    print(f"负样本匹配总数: {len(impostor_scores)}")
    
    # 计算FAR和FRR
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    # FRR = 正样本被拒绝的比例
    frr = np.sum(genuine_scores < threshold) / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
    
    # FAR = 负样本被接受的比例
    far = np.sum(impostor_scores >= threshold) / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
    
    return far, frr, genuine_scores.tolist(), impostor_scores.tolist()


def plot_roc_curve(genuine_scores: List[float], 
                   impostor_scores: List[float],
                   save_path: str = 'roc_curve.png'):
    """
    绘制ROC曲线
    
    Args:
        genuine_scores: 正样本分数
        impostor_scores: 负样本分数
        save_path: 保存路径
    """
    thresholds = np.linspace(0, 1, 100)
    frr_list = []
    far_list = []
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    for threshold in thresholds:
        frr = np.sum(genuine_scores < threshold) / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
        far = np.sum(impostor_scores >= threshold) / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
        frr_list.append(frr)
        far_list.append(far)
    
    # 计算EER（Equal Error Rate）
    eer_idx = np.argmin(np.abs(np.array(frr_list) - np.array(far_list)))
    eer = (frr_list[eer_idx] + far_list[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(far_list, [1 - frr for frr in frr_list], 'b-', linewidth=2, label='ROC曲线')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='随机分类器')
    plt.plot(far_list[eer_idx], 1 - frr_list[eer_idx], 'ro', markersize=10, label=f'EER={eer:.4f}')
    plt.xlabel('FAR (False Accept Rate)', fontsize=12)
    plt.ylabel('1 - FRR (True Accept Rate)', fontsize=12)
    plt.title('ROC曲线', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nEER (Equal Error Rate): {eer:.4f} (阈值: {eer_threshold:.4f})")
    
    return eer, eer_threshold


def plot_score_distribution(genuine_scores: List[float],
                           impostor_scores: List[float],
                           save_path: str = 'score_distribution.png'):
    """
    绘制分数分布直方图
    
    Args:
        genuine_scores: 正样本分数
        impostor_scores: 负样本分数
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(genuine_scores, bins=50, alpha=0.7, label='正样本 (Genuine)', color='green', density=True)
    plt.hist(impostor_scores, bins=50, alpha=0.7, label='负样本 (Impostor)', color='red', density=True)
    
    plt.xlabel('匹配分数', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.title('匹配分数分布', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """主函数"""
    # 配置路径
    dataset_dir = '../datasets/db4_b'
    output_dir = '../output/20251027-172033/2'  # 包含mnt文件的目录
    template_dir = './templates_db4_b'
    
    # 检查路径
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集目录不存在: {dataset_dir}")
        return
    
    if not os.path.exists(output_dir):
        print(f"错误: 输出目录不存在: {output_dir}")
        print("请先运行FingerNet提取细节点")
        return
    
    # 创建模板匹配器（使用优化后的参数）
    matcher = TemplateMatcher(
        k_neighbors=5,
        template_radius=100.0,
        distance_tolerance=15.0,
        angle_tolerance=30.0,
        orientation_tolerance=30.0,
        max_distance=50.0,
        match_threshold=0.65  # 平衡的匹配阈值
    )
    
    # 查找所有文件
    print("查找数据集文件...")
    files = find_mnt_files(dataset_dir, output_dir)
    print(f"找到 {len(files)} 个文件")
    
    if len(files) == 0:
        print("错误: 未找到任何文件")
        return
    
    # 构建并保存模板
    build_templates_and_save(matcher, files, template_dir)
    
    # 计算FAR和FRR
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n" + "="*60)
    print("在不同阈值下计算FAR和FRR:")
    print("="*60)
    
    results = []
    all_genuine_scores = []
    all_impostor_scores = []
    
    for threshold in thresholds:
        far, frr, genuine_scores, impostor_scores = compute_far_frr(
            matcher, files, template_dir, threshold
        )
        
        results.append((threshold, far, frr))
        all_genuine_scores = genuine_scores
        all_impostor_scores = impostor_scores
        
        print(f"\n阈值: {threshold:.2f}")
        print(f"  FAR (False Accept Rate): {far:.4f} ({far*100:.2f}%)")
        print(f"  FRR (False Reject Rate): {frr:.4f} ({frr*100:.2f}%)")
        print(f"  准确率: {(1 - far - frr) / 2 + 0.5:.4f}")
    
    # 绘制ROC曲线
    print("\n绘制ROC曲线...")
    eer, eer_threshold = plot_roc_curve(all_genuine_scores, all_impostor_scores)
    
    # 绘制分数分布
    print("绘制分数分布...")
    plot_score_distribution(all_genuine_scores, all_impostor_scores)
    
    # 输出统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print("="*60)
    print(f"正样本数量: {len(all_genuine_scores)}")
    print(f"负样本数量: {len(all_impostor_scores)}")
    print(f"正样本平均分数: {np.mean(all_genuine_scores):.4f}")
    print(f"负样本平均分数: {np.mean(all_impostor_scores):.4f}")
    print(f"正样本标准差: {np.std(all_genuine_scores):.4f}")
    print(f"负样本标准差: {np.std(all_impostor_scores):.4f}")
    print(f"\n最佳阈值 (EER): {eer_threshold:.4f}")
    print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
    
    # 保存结果
    result_file = 'evaluation_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("指纹匹配算法评估结果\n")
        f.write("="*60 + "\n\n")
        f.write(f"数据集: db4_b\n")
        f.write(f"正样本数量: {len(all_genuine_scores)}\n")
        f.write(f"负样本数量: {len(all_impostor_scores)}\n\n")
        
        f.write("不同阈值下的FAR和FRR:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'阈值':<10} {'FAR':<15} {'FRR':<15} {'准确率':<15}\n")
        f.write("-"*60 + "\n")
        for threshold, far, frr in results:
            accuracy = (1 - far - frr) / 2 + 0.5
            f.write(f"{threshold:<10.2f} {far:<15.4f} {frr:<15.4f} {accuracy:<15.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write(f"最佳阈值 (EER): {eer_threshold:.4f}\n")
        f.write(f"EER: {eer:.4f} ({eer*100:.2f}%)\n")
    
    print(f"\n结果已保存到: {result_file}")
    print("ROC曲线已保存到: roc_curve.png")
    print("分数分布已保存到: score_distribution.png")


if __name__ == "__main__":
    main()

