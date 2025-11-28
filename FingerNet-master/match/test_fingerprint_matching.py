"""
指纹匹配算法评估脚本
支持命令行参数配置，支持多种图像格式（包括TIF）
"""

import os
import sys
import argparse
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加match模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from match import TemplateMatcher, Minutia, MatchResult

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='指纹匹配算法评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法（输出目录和模板目录自动生成）
  # 输出目录: ./evaluation_results_fvc2004_DB1_A
  # 模板目录: ./templates_fvc2004_DB1_A
  python test_fingerprint_matching.py --dataset_dir ../datasets/fvc2004_DB1_A --mnt_dir ../output/20251111-172714/1
  
  # 指定自定义模板目录（覆盖自动生成的目录名）
  python test_fingerprint_matching.py --dataset_dir ../datasets/fvc2000_db4_b --mnt_dir ../output/20251027-172033/2 --template_dir ./my_templates
  
  # 指定图像格式
  python test_fingerprint_matching.py --dataset_dir ../datasets/fvc2000_db4_b --mnt_dir ../output/20251027-172033/2 --image_format .tif
  
  # 指定自定义输出目录（覆盖自动生成的目录名）
  python test_fingerprint_matching.py --dataset_dir ../datasets/fvc2004_DB1_A --mnt_dir ../output/20251111-172714/1 --output_dir ./my_results
        """
    )
    
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='数据集目录（包含图像文件，如.bmp, .png, .tif等）')
    parser.add_argument('--mnt_dir', type=str, required=True,
                        help='输出目录（包含.mnt文件的目录）')
    parser.add_argument('--image_format', type=str, default=None,
                        help='图像格式（如.bmp, .png, .tif），None表示自动检测（默认: None）')
    parser.add_argument('--match_threshold', type=float, default=0.65,
                        help='匹配阈值（默认: 0.65）')
    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='局部模板中使用的最近邻数量（默认: 5）')
    parser.add_argument('--template_radius', type=float, default=100.0,
                        help='模板搜索半径（像素）（默认: 100.0）')
    parser.add_argument('--distance_tolerance', type=float, default=15.0,
                        help='距离容差（像素）（默认: 15.0）')
    parser.add_argument('--angle_tolerance', type=float, default=30.0,
                        help='角度容差（度）（默认: 30.0）')
    parser.add_argument('--orientation_tolerance', type=float, default=30.0,
                        help='方向容差（度）（默认: 30.0）')
    parser.add_argument('--max_distance', type=float, default=40.0,
                        help='RANSAC中最大匹配距离（像素）（默认: 40.0）')
    parser.add_argument('--ransac_runs', type=int, default=3,
                        help='RANSAC运行次数，多次运行取最佳结果以提高稳定性（默认: 3）')
    parser.add_argument('--thresholds', type=str, default='0.3,0.4,0.5,0.6,0.7',
                        help='评估阈值列表，逗号分隔（默认: 0.3,0.4,0.5,0.6,0.7）')
    parser.add_argument('--templates_per_finger', type=int, default=1,
                        help='每个手指用于模板的样本数量（固定 enrollment，默认: 1）')
    parser.add_argument('--rebuild_templates', action='store_true',
                        help='强制重新构建模板（即使模板文件已存在）')
    parser.add_argument('--num_visualizations', type=int, default=5,
                        help='每种类型生成的可视化图片数量（默认: 5）')
    
    return parser.parse_args()


def find_mnt_files(dataset_dir: str, output_dir: str, image_format: str = None) -> List[Tuple[str, str, str, str]]:
    """
    查找所有图像和对应的mnt文件
    
    Args:
        dataset_dir: 数据集目录（包含图像文件）
        output_dir: 输出目录（包含.mnt文件）
        image_format: 图像格式（如'.bmp', '.tif'），None表示自动检测
        
    Returns:
        [(finger_id, sample_id, image_path, mnt_path), ...]
    """
    files = []
    
    # 支持的图像格式（包括TIF）
    supported_formats = ['.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
    
    # 如果指定了格式，优先使用
    if image_format:
        if not image_format.startswith('.'):
            image_format = '.' + image_format
        supported_formats = [image_format] + [f for f in supported_formats if f != image_format]
    
    # 查找图像文件
    image_files = []
    detected_format = None
    
    for fmt in supported_formats:
        pattern = '*' + fmt
        import glob
        found_files = glob.glob(os.path.join(dataset_dir, pattern))
        if len(found_files) > 0:
            image_files = found_files
            detected_format = fmt
            print(f"检测到图像格式: {detected_format} ({len(image_files)} 个文件)")
            break
    
    if len(image_files) == 0:
        print(f"警告: 在 {dataset_dir} 中未找到支持的图像文件")
        return files
    
    # 解析文件名并查找对应的mnt文件
    for image_file in sorted(image_files):
        # 获取文件名（不含扩展名）
        base_name = os.path.basename(image_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # 解析文件名：00000_00 -> finger_id=00000, sample_id=00
        name_parts = name_without_ext.split('_')
        if len(name_parts) >= 2:
            finger_id = name_parts[0]
            sample_id = '_'.join(name_parts[1:])  # 支持多段sample_id
            
            image_path = image_file
            mnt_path = os.path.join(output_dir, name_without_ext + '.mnt')
            
            # 检查mnt文件是否存在
            if os.path.exists(mnt_path):
                files.append((finger_id, sample_id, image_path, mnt_path))
            else:
                print(f"警告: 未找到mnt文件: {mnt_path}")
    
    return files


def build_templates_and_save(matcher: TemplateMatcher, 
                            files: List[Tuple[str, str, str, str]],
                            template_dir: str,
                            rebuild: bool = False,
                            templates_per_finger: int = 1) -> float:
    """
    为所有指纹构建模板并保存（多模板策略）
    
    Args:
        matcher: 模板匹配器
        files: 文件列表
        template_dir: 模板保存目录
        rebuild: 是否强制重新构建模板
        templates_per_finger: 每个手指用于模板的样本数量
        
    Returns:
        模板构建总时间（秒）
    """
    os.makedirs(template_dir, exist_ok=True)
    
    print("="*60)
    print("构建并保存多模板（每个手指包含固定数量的 enrollment 样本）...")
    print("="*60)
    
    build_start_time = time.time()
    
    # 按手指ID分组
    finger_groups = {}
    for finger_id, sample_id, image_path, mnt_path in files:
        if finger_id not in finger_groups:
            finger_groups[finger_id] = []
        finger_groups[finger_id].append((sample_id, image_path, mnt_path))
    
    templates_per_finger = max(1, templates_per_finger)
    # 为每个手指构建包含固定数量样本的多模板
    for finger_id in sorted(finger_groups.keys()):
        template_file = os.path.join(template_dir, f"{finger_id}_multi.pkl")
        
        need_rebuild = rebuild
        desired_count = min(templates_per_finger, len(finger_groups[finger_id]))
        if os.path.exists(template_file) and not rebuild:
            existing_templates, _, _ = matcher.load_multi_template(template_file, verbose=False)
            existing_count = len(existing_templates)
            if existing_count != desired_count:
                print(f"检测到 {finger_id} 的模板数量({existing_count})与期望({desired_count})不一致，重新构建...")
                need_rebuild = True
            else:
                print(f"跳过 {finger_id}: 模板已存在且数量匹配（使用 --rebuild_templates 可强制重建）")
        if os.path.exists(template_file) and not need_rebuild:
            continue
        
        samples = sorted(finger_groups[finger_id], key=lambda x: x[0])
        enroll_samples = samples[:desired_count]
        multi_templates = []
        total_minutiae = 0
        
        sample_ids = []  # 只保存有效样本的ID
        for sample_id, image_path, mnt_path in enroll_samples:
            # 加载细节点
            minutiae = matcher.load_minutiae_from_mnt(mnt_path)
            
            if len(minutiae) == 0:
                print(f"跳过 {finger_id}_{sample_id}: 无细节点")
                continue
            
            # 构建模板
            templates = matcher.compute_templates(minutiae)
            multi_templates.append((templates, minutiae))
            sample_ids.append(f"{finger_id}_{sample_id}")  # 只添加有效样本的ID
            total_minutiae += len(minutiae)
        
        if len(multi_templates) == 0:
            print(f"跳过 {finger_id}: 无有效样本")
            continue
        
        # 保存多模板（包含样本ID，用于避免数据泄露）
        metadata = {
            'finger_id': finger_id,
            'num_samples': len(multi_templates),
            'total_minutiae': total_minutiae
        }
        matcher.save_multi_template(template_file, multi_templates, metadata, sample_ids)
        
        print(f"已保存多模板: {finger_id} (包含 {len(multi_templates)} 个样本, 共 {total_minutiae} 个细节点)")
    
    build_time = time.time() - build_start_time
    print(f"\n模板构建总时间: {build_time:.3f} 秒")
    return build_time


def compute_far_frr(matcher: TemplateMatcher,
                    files: List[Tuple[str, str, str, str]],
                    template_dir: str,
                    threshold: float = 0.5) -> Tuple[float, float, List[float], List[float], Dict[str, List[Dict[str, Any]]], float, Dict[str, Any]]:
    """
    计算FAR和FRR
    
    Args:
        matcher: 模板匹配器
        files: 文件列表
        template_dir: 模板目录
        threshold: 匹配阈值（用于计算FAR/FRR，不影响匹配过程）
        
    Returns:
        (FAR, FRR, genuine_scores, impostor_scores, sample_details, matching_time, match_stats)
        match_stats包含: total_matches, genuine_matches, impostor_matches, avg_match_time, avg_genuine_time, avg_impostor_time
    """
    print("="*60)
    print("计算FAR和FRR...")
    print("="*60)
    
    matching_start_time = time.time()
    
    genuine_scores = []  # 正样本分数（同一手指不同样本）
    impostor_scores = []  # 负样本分数（不同手指）
    sample_details: Dict[str, List[Dict[str, Any]]] = {
        'genuine': [],
        'impostor': []
    }
    
    # 按手指ID分组
    finger_groups = {}
    sample_path_map: Dict[str, Tuple[str, str]] = {}
    for finger_id, sample_id, image_path, mnt_path in files:
        if finger_id not in finger_groups:
            finger_groups[finger_id] = []
        finger_groups[finger_id].append((sample_id, image_path, mnt_path))
        sample_path_map[f"{finger_id}_{sample_id}"] = (image_path, mnt_path)
    
    finger_ids = sorted(finger_groups.keys())
    num_fingers = len(finger_ids)
    
    print(f"找到 {num_fingers} 个手指，每个手指有多个样本")
    
    # 计算正样本分数（Genuine）
    # 正样本：同一手指的查询样本与固定 enrollment 模板的匹配
    print("\n计算正样本分数（查询样本与固定 enrollment 模板的匹配）...")
    genuine_count = 0
    
    # 加载所有手指的 enrollment 样本ID
    multi_template_sample_ids: Dict[str, List[str]] = {}
    for finger_id in finger_ids:
        multi_template_file = os.path.join(template_dir, f"{finger_id}_multi.pkl")
        if os.path.exists(multi_template_file):
            _, _, sample_ids = matcher.load_multi_template(multi_template_file, verbose=False)
            multi_template_sample_ids[finger_id] = sample_ids

    # 多模板策略：使用多模板进行匹配（固定 enrollment 样本）
    print("使用多模板策略进行匹配（固定模板样本）...")
    genuine_match_start_time = time.time()
    for finger_id in finger_ids:
        samples = finger_groups[finger_id]
        if len(samples) < 2:
            continue
        
        # 加载多模板
        multi_template_file = os.path.join(template_dir, f"{finger_id}_multi.pkl")
        if not os.path.exists(multi_template_file):
            continue
        enrolled_ids = set(multi_template_sample_ids.get(finger_id, []))
        
        # 仅使用未纳入模板的样本作为查询指纹
        query_samples = [
            (sample_id, image_path, mnt_path)
            for sample_id, image_path, mnt_path in samples
            if f"{finger_id}_{sample_id}" not in enrolled_ids
        ]
        if len(query_samples) == 0:
            continue
        
        for sample_id, image_path, mnt_path in query_samples:
            query_sample_id = f"{finger_id}_{sample_id}"
            query_minutiae = matcher.load_minutiae_from_mnt(mnt_path)
            if len(query_minutiae) == 0:
                continue
            
            result = matcher.match_with_stored_template(
                multi_template_file, 
                query_minutiae,
                exclude_query_sample_id=None
            )
            genuine_scores.append(result.score)
            genuine_count += 1

            matched_sample_id = result.metadata.get('stored_sample_id') if result.metadata else None
            # 如果匹配失败，使用第一个 enrollment 样本作为参考
            if matched_sample_id is None or matched_sample_id not in sample_path_map:
                enrolled_ids = multi_template_sample_ids.get(finger_id, [])
                matched_sample_id = enrolled_ids[0] if enrolled_ids else None
            ref_image_path, ref_mnt_path = sample_path_map.get(matched_sample_id, (None, None))
            sample_details['genuine'].append({
                'score': result.score,
                'num_matches': result.num_matches,
                'inlier_ratio': result.inlier_ratio,
                'matched_pairs': result.matched_pairs,  # 保存匹配对用于可视化
                'finger_id': finger_id,
                'query_sample_id': query_sample_id,
                'reference_sample_id': matched_sample_id,
                'query_image_path': image_path,
                'query_mnt_path': mnt_path,
                'reference_image_path': ref_image_path,
                'reference_mnt_path': ref_mnt_path,
                'metadata': result.metadata or {}
            })
            
            if genuine_count % 50 == 0:
                print(f"  已处理 {genuine_count} 个正样本匹配...")
    
    genuine_match_time = time.time() - genuine_match_start_time
    print(f"正样本匹配总数: {len(genuine_scores)}")
    
    # 计算负样本分数（Impostor）
    # 负样本：不同手指的查询样本与 enrollment 模板的匹配
    print("\n计算负样本分数（不同手指的查询样本与 enrollment 模板的匹配）...")
    impostor_count = 0
    
    # 多模板策略：使用多模板进行匹配（负样本不需要排除，因为不同手指）
    impostor_match_start_time = time.time()
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
            for sample_id2, image_path2, mnt_path2 in samples2:
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

                matched_sample_id = result.metadata.get('stored_sample_id') if result.metadata else None
                # 如果匹配失败，使用第一个 enrollment 样本作为参考
                if matched_sample_id is None or matched_sample_id not in sample_path_map:
                    enrolled_ids = multi_template_sample_ids.get(finger_id1, [])
                    matched_sample_id = enrolled_ids[0] if enrolled_ids else None
                ref_image_path, ref_mnt_path = sample_path_map.get(matched_sample_id, (None, None))
                sample_details['impostor'].append({
                    'score': result.score,
                    'num_matches': result.num_matches,
                    'inlier_ratio': result.inlier_ratio,
                    'matched_pairs': result.matched_pairs,  # 保存匹配对用于可视化
                    'finger_id': finger_id1,
                    'query_sample_id': f"{finger_id2}_{sample_id2}",
                    'reference_sample_id': matched_sample_id,
                    'query_image_path': image_path2,
                'query_mnt_path': mnt_path2,
                'reference_image_path': ref_image_path,
                'reference_mnt_path': ref_mnt_path,
                'metadata': result.metadata or {}
            })
                
                if impostor_count % 100 == 0:
                    print(f"  已处理 {impostor_count} 个负样本匹配...")
    
    impostor_match_time = time.time() - impostor_match_start_time
    print(f"负样本匹配总数: {len(impostor_scores)}")
    
    # 计算FAR和FRR
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    # FRR = 正样本被拒绝的比例
    frr = np.sum(genuine_scores < threshold) / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
    
    # FAR = 负样本被接受的比例
    far = np.sum(impostor_scores >= threshold) / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
    
    matching_time = time.time() - matching_start_time
    
    # 计算平均匹配时间（分别统计正样本和负样本的时间）
    total_matches = len(genuine_scores) + len(impostor_scores)
    avg_match_time = matching_time / total_matches if total_matches > 0 else 0.0
    avg_genuine_time = genuine_match_time / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
    avg_impostor_time = impostor_match_time / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
    
    print(f"\n匹配计算总时间: {matching_time:.3f} 秒")
    print(f"总匹配次数: {total_matches} (正样本: {len(genuine_scores)}, 负样本: {len(impostor_scores)})")
    print(f"平均每次匹配时间: {avg_match_time*1000:.3f} 毫秒")
    print(f"平均正样本匹配时间: {avg_genuine_time*1000:.3f} 毫秒")
    print(f"平均负样本匹配时间: {avg_impostor_time*1000:.3f} 毫秒")
    
    return far, frr, genuine_scores.tolist(), impostor_scores.tolist(), sample_details, matching_time, {
        'total_matches': total_matches,
        'genuine_matches': len(genuine_scores),
        'impostor_matches': len(impostor_scores),
        'genuine_match_time': genuine_match_time,
        'impostor_match_time': impostor_match_time,
        'avg_match_time': avg_match_time,
        'avg_genuine_time': avg_genuine_time,
        'avg_impostor_time': avg_impostor_time
    }


def _compute_rate_metrics(genuine_scores: np.ndarray,
                          impostor_scores: np.ndarray,
                          thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算不同阈值下的 FAR、FRR、Precision、Recall
    """
    total_genuine = len(genuine_scores)
    total_impostor = len(impostor_scores)

    far_list = []
    frr_list = []
    precision_list = []
    recall_list = []

    for threshold in thresholds:
        tp = np.sum(genuine_scores >= threshold)
        fn = total_genuine - tp
        fp = np.sum(impostor_scores >= threshold)
        tn = total_impostor - fp

        far = fp / total_impostor if total_impostor > 0 else 0.0
        frr = fn / total_genuine if total_genuine > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / total_genuine if total_genuine > 0 else 0.0

        far_list.append(far)
        frr_list.append(frr)
        precision_list.append(precision)
        recall_list.append(recall)

    return (np.array(far_list),
            np.array(frr_list),
            np.array(precision_list),
            np.array(recall_list))


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


def plot_far_frr_curve(genuine_scores: List[float],
                       impostor_scores: List[float],
                       save_path: str = 'far_frr_curve.png'):
    """
    绘制阈值 vs FAR/FRR 曲线
    """
    thresholds = np.linspace(0, 1, 100)
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    far_list, frr_list, _, _ = _compute_rate_metrics(genuine_scores, impostor_scores, thresholds)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, far_list, 'r-', linewidth=2, label='FAR')
    plt.plot(thresholds, frr_list, 'b-', linewidth=2, label='FRR')
    plt.xlabel('阈值', fontsize=12)
    plt.ylabel('错误率', fontsize=12)
    plt.title('阈值与 FAR/FRR 曲线', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_det_curve(genuine_scores: List[float],
                   impostor_scores: List[float],
                   save_path: str = 'det_curve.png'):
    """
    绘制 DET 曲线（FAR vs FRR，采用对数刻度）
    """
    thresholds = np.linspace(0, 1, 200)
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    far_list, frr_list, _, _ = _compute_rate_metrics(genuine_scores, impostor_scores, thresholds)

    # 避免出现 0 导致对数刻度报错
    far_list = np.clip(far_list, 1e-6, 1.0)
    frr_list = np.clip(frr_list, 1e-6, 1.0)

    plt.figure(figsize=(8, 6))
    plt.plot(far_list, frr_list, 'm-', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('FAR (False Accept Rate)', fontsize=12)
    plt.ylabel('FRR (False Reject Rate)', fontsize=12)
    plt.title('DET 曲线', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_precision_recall_curve(genuine_scores: List[float],
                                impostor_scores: List[float],
                                save_path: str = 'precision_recall_curve.png'):
    """
    绘制精确率-召回率曲线
    """
    thresholds = np.linspace(0, 1, 200)
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    _, _, precision_list, recall_list = _compute_rate_metrics(genuine_scores, impostor_scores, thresholds)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_list, precision_list, 'g-', linewidth=2)
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('Precision-Recall 曲线', fontsize=14)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def render_match_record(matcher: TemplateMatcher,
                        record: Dict[str, Any],
                        output_path: str) -> bool:
    """
    根据记录渲染匹配可视化图
    
    使用记录中保存的匹配结果信息，而不是重新匹配，确保可视化显示的分数与评估时一致
    """
    query_image_path = record.get('query_image_path')
    reference_image_path = record.get('reference_image_path')
    query_mnt_path = record.get('query_mnt_path')
    reference_mnt_path = record.get('reference_mnt_path')

    if not all([query_image_path, reference_image_path, query_mnt_path, reference_mnt_path]):
        print(f"跳过可视化：记录缺少路径信息，record={record}")
        return False

    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if query_image is None or reference_image is None:
        print(f"跳过可视化：无法读取图像文件 -> {query_image_path}, {reference_image_path}")
        return False

    query_minutiae = matcher.load_minutiae_from_mnt(query_mnt_path)
    reference_minutiae = matcher.load_minutiae_from_mnt(reference_mnt_path)
    if len(query_minutiae) == 0 or len(reference_minutiae) == 0:
        print(f"跳过可视化：细节点为空 -> {query_mnt_path}, {reference_mnt_path}")
        return False

    # 使用记录中保存的匹配结果信息，而不是重新匹配
    # 这样可以确保可视化显示的分数与评估时一致
    # 从记录中获取匹配结果信息
    score = record.get('score', 0.0)
    num_matches = record.get('num_matches', 0)
    inlier_ratio = record.get('inlier_ratio', 0.0)
    matched_pairs = record.get('matched_pairs', [])
    
    # 创建 MatchResult 对象用于可视化
    result = MatchResult(
        score=score,
        num_matches=num_matches,
        transform_matrix=np.eye(3),  # 可视化不需要变换矩阵
        matched_pairs=matched_pairs,
        inlier_ratio=inlier_ratio,
        metadata=record.get('metadata', {})
    )
    
    matcher.visualize_matches(
        reference_image,
        reference_minutiae,
        query_image,
        query_minutiae,
        result,
        output_path,
        image1_label=os.path.basename(reference_image_path),
        image2_label=os.path.basename(query_image_path)
    )
    return True


def generate_match_visualizations(matcher: TemplateMatcher,
                                  genuine_records: List[Dict[str, Any]],
                                  impostor_records: List[Dict[str, Any]],
                                  threshold: float,
                                  success_dir: str,
                                  false_negative_dir: str,
                                  false_positive_dir: str,
                                  num_visualizations: int = 5) -> Tuple[Dict[str, List[str]], float]:
    """
    基于记录生成代表性的匹配可视化
    
    Args:
        matcher: 模板匹配器
        genuine_records: 正样本匹配记录列表
        impostor_records: 负样本匹配记录列表
        threshold: 匹配阈值
        success_dir: 成功匹配可视化输出目录
        false_negative_dir: 假阴性（本该成功却失败）可视化输出目录
        false_positive_dir: 假阳性（本该失败却成功）可视化输出目录
        num_visualizations: 每种类型生成的可视化图片数量
        
    Returns:
        (包含生成文件路径的字典, 可视化生成时间)
    """
    visualization_start_time = time.time()
    
    outputs: Dict[str, List[str]] = {
        'success': [],
        'false_negative': [],
        'false_positive': []
    }

    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(false_negative_dir, exist_ok=True)
    os.makedirs(false_positive_dir, exist_ok=True)

    # 1. 成功案例（分数高于阈值，按分数从高到低排序）
    success_candidates = [
        rec for rec in genuine_records
        if rec.get('score', 0.0) >= threshold
        and rec.get('query_image_path') and rec.get('reference_image_path')
    ]
    if len(genuine_records) > 0:
        print(f"  正样本记录总数: {len(genuine_records)}")
        print(f"  成功候选数（分数 >= {threshold}）: {len(success_candidates)}")
        scores_above_threshold = [r.get('score', 0.0) for r in genuine_records if r.get('score', 0.0) >= threshold]
        if len(scores_above_threshold) > 0:
            print(f"  成功候选分数范围: {min(scores_above_threshold):.4f} - {max(scores_above_threshold):.4f}")
    if success_candidates:
        # 按分数从高到低排序
        success_candidates.sort(key=lambda r: r['score'], reverse=True)
        # 取前 num_visualizations 个
        top_successes = success_candidates[:num_visualizations]
        for idx, record in enumerate(top_successes, 1):
            output_path = os.path.join(success_dir, f'match_success_top{idx}.png')
            if render_match_record(matcher, record, output_path):
                outputs['success'].append(output_path)
                print(f"  生成成功案例 {idx}: {os.path.basename(output_path)} (分数: {record['score']:.4f})")

    # 2. 假阴性案例（本该成功却失败，分数低于阈值，按分数从低到高排序）
    false_negative_candidates = [
        rec for rec in genuine_records
        if rec.get('score', 0.0) < threshold
        and rec.get('query_image_path') and rec.get('reference_image_path')
    ]
    if len(genuine_records) > 0:
        print(f"  假阴性候选数（分数 < {threshold}）: {len(false_negative_candidates)}")
        scores_below_threshold = [r.get('score', 0.0) for r in genuine_records if r.get('score', 0.0) < threshold]
        if len(scores_below_threshold) > 0:
            print(f"  假阴性候选分数范围: {min(scores_below_threshold):.4f} - {max(scores_below_threshold):.4f}")
    if false_negative_candidates:
        # 按分数从低到高排序（最差的在前）
        false_negative_candidates.sort(key=lambda r: r['score'])
        # 取前 num_visualizations 个
        top_false_negatives = false_negative_candidates[:num_visualizations]
        for idx, record in enumerate(top_false_negatives, 1):
            output_path = os.path.join(false_negative_dir, f'match_false_negative_{idx}.png')
            if render_match_record(matcher, record, output_path):
                outputs['false_negative'].append(output_path)
                print(f"  生成假阴性案例 {idx}: {os.path.basename(output_path)} (分数: {record['score']:.4f})")

    # 3. 假阳性案例（本该失败却成功，分数高于阈值，按分数从高到低排序）
    false_positive_candidates = [
        rec for rec in impostor_records
        if rec.get('score', 0.0) >= threshold
        and rec.get('query_image_path') and rec.get('reference_image_path')
    ]
    if len(impostor_records) > 0:
        print(f"  负样本记录总数: {len(impostor_records)}")
        print(f"  假阳性候选数（分数 >= {threshold}）: {len(false_positive_candidates)}")
        scores_above_threshold = [r.get('score', 0.0) for r in impostor_records if r.get('score', 0.0) >= threshold]
        if len(scores_above_threshold) > 0:
            print(f"  假阳性候选分数范围: {min(scores_above_threshold):.4f} - {max(scores_above_threshold):.4f}")
        else:
            max_impostor_score = max([r.get('score', 0.0) for r in impostor_records]) if impostor_records else 0.0
            print(f"  负样本最高分数: {max_impostor_score:.4f} (低于阈值 {threshold})")
    if false_positive_candidates:
        # 按分数从高到低排序（最严重的在前）
        false_positive_candidates.sort(key=lambda r: r['score'], reverse=True)
        # 取前 num_visualizations 个
        top_false_positives = false_positive_candidates[:num_visualizations]
        for idx, record in enumerate(top_false_positives, 1):
            output_path = os.path.join(false_positive_dir, f'match_false_positive_{idx}.png')
            if render_match_record(matcher, record, output_path):
                outputs['false_positive'].append(output_path)
                print(f"  生成假阳性案例 {idx}: {os.path.basename(output_path)} (分数: {record['score']:.4f})")

    visualization_time = time.time() - visualization_start_time
    print(f"\n可视化生成总时间: {visualization_time:.3f} 秒")
    
    return outputs, visualization_time


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
    
    plt.hist(genuine_scores, bins=200, alpha=0.7, label='正样本 (Genuine)', color='green', density=True)
    plt.hist(impostor_scores, bins=200, alpha=0.7, label='负样本 (Impostor)', color='red', density=True)
    
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
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查路径
    if not os.path.exists(args.dataset_dir):
        print(f"错误: 数据集目录不存在: {args.dataset_dir}")
        return
    
    if not os.path.exists(args.mnt_dir):
        print(f"错误: mnt文件目录不存在: {args.mnt_dir}")
        print("请先运行FingerNet提取细节点")
        return
    
    # 从数据集目录路径中提取数据集名称
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir)) or "dataset"
    
    # 根据数据集名称自动生成输出目录和模板目录
    output_dir = f'./evaluation_results_{dataset_name}'
    template_dir = f'./templates_{dataset_name}'
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    print(f"模板目录: {template_dir}")
    
    # 创建模板匹配器（使用参数配置）
    matcher = TemplateMatcher(
        k_neighbors=args.k_neighbors,
        template_radius=args.template_radius,
        distance_tolerance=args.distance_tolerance,
        angle_tolerance=args.angle_tolerance,
        orientation_tolerance=args.orientation_tolerance,
        max_distance=args.max_distance,
        match_threshold=args.match_threshold,
        ransac_runs=args.ransac_runs,
        templates_per_finger=args.templates_per_finger
    )
    
    # 查找所有文件
    print("="*60)
    print("查找数据集文件...")
    print(f"数据集目录: {args.dataset_dir}")
    print(f"mnt文件目录: {args.mnt_dir}")
    print(f"图像格式: {args.image_format if args.image_format else '自动检测'}")
    print("="*60)
    
    files = find_mnt_files(args.dataset_dir, args.mnt_dir, args.image_format)
    print(f"找到 {len(files)} 个文件")
    
    if len(files) == 0:
        print("错误: 未找到任何文件")
        return
    
    # 构建并保存模板
    template_build_time = build_templates_and_save(
        matcher,
        files,
        template_dir,
        rebuild=args.rebuild_templates,
        templates_per_finger=args.templates_per_finger
    )
    
    # 解析阈值列表
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    
    print("\n" + "="*60)
    print("在不同阈值下计算FAR和FRR:")
    print("="*60)
    
    results = []
    all_genuine_scores = []
    all_impostor_scores = []
    
    # 只计算一次分数（所有阈值共享相同的分数列表）
    # 使用第一个阈值计算分数，后续只用于计算FAR/FRR
    _, _, all_genuine_scores, all_impostor_scores, sample_details, matching_time, match_stats = compute_far_frr(
        matcher, files, template_dir, thresholds[0]
    )
    
    # 使用计算好的分数，在不同阈值下计算FAR和FRR
    genuine_scores_array = np.array(all_genuine_scores)
    impostor_scores_array = np.array(all_impostor_scores)
    
    for threshold in thresholds:
        # 使用已计算的分数计算FAR和FRR
        frr = np.sum(genuine_scores_array < threshold) / len(genuine_scores_array) if len(genuine_scores_array) > 0 else 0.0
        far = np.sum(impostor_scores_array >= threshold) / len(impostor_scores_array) if len(impostor_scores_array) > 0 else 0.0
        results.append((threshold, far, frr))
        
        print(f"\n阈值: {threshold:.2f}")
        print(f"  FAR (False Accept Rate): {far:.4f} ({far*100:.2f}%)")
        print(f"  FRR (False Reject Rate): {frr:.4f} ({frr*100:.2f}%)")
        print(f"  准确率: {(1 - far - frr) / 2 + 0.5:.4f}")
    
    # 绘制各种曲线图（统计绘图时间）
    plot_start_time = time.time()
    print("\n绘制各种评估曲线图...")
    
    roc_path = os.path.join(output_dir, 'evaluation_roc_curve.png')
    eer, eer_threshold = plot_roc_curve(all_genuine_scores, all_impostor_scores, roc_path)
    
    dist_path = os.path.join(output_dir, 'evaluation_score_distribution.png')
    plot_score_distribution(all_genuine_scores, all_impostor_scores, dist_path)

    far_frr_path = os.path.join(output_dir, 'evaluation_far_frr_curve.png')
    plot_far_frr_curve(all_genuine_scores, all_impostor_scores, far_frr_path)

    det_path = os.path.join(output_dir, 'evaluation_det_curve.png')
    plot_det_curve(all_genuine_scores, all_impostor_scores, det_path)

    pr_path = os.path.join(output_dir, 'evaluation_precision_recall_curve.png')
    plot_precision_recall_curve(all_genuine_scores, all_impostor_scores, pr_path)
    
    plot_time = time.time() - plot_start_time
    print(f"评估曲线图绘制时间: {plot_time:.3f} 秒")

    # 生成代表性匹配可视化
    print("\n" + "="*60)
    print("生成匹配可视化图...")
    print("="*60)
    success_dir = os.path.join(output_dir, 'success_matches')
    false_negative_dir = os.path.join(output_dir, 'false_negative_matches')
    false_positive_dir = os.path.join(output_dir, 'false_positive_matches')
    visualization_outputs, visualization_time = generate_match_visualizations(
        matcher,
        sample_details.get('genuine', []),
        sample_details.get('impostor', []),
        args.match_threshold,
        success_dir,
        false_negative_dir,
        false_positive_dir,
        args.num_visualizations
    )
    
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
    print(f"\n时间统计:")
    print(f"  模板构建时间: {template_build_time:.3f} 秒")
    print(f"  匹配计算时间: {matching_time:.3f} 秒")
    print(f"    总匹配次数: {match_stats['total_matches']} (正样本: {match_stats['genuine_matches']}, 负样本: {match_stats['impostor_matches']})")
    print(f"    平均每次匹配时间: {match_stats['avg_match_time']*1000:.3f} 毫秒")
    print(f"    平均正样本匹配时间: {match_stats['avg_genuine_time']*1000:.3f} 毫秒")
    print(f"    平均负样本匹配时间: {match_stats['avg_impostor_time']*1000:.3f} 毫秒")
    print(f"  评估曲线图绘制时间: {plot_time:.3f} 秒")
    print(f"  可视化生成时间: {visualization_time:.3f} 秒")
    total_time = template_build_time + matching_time + plot_time + visualization_time
    print(f"  总时间: {total_time:.3f} 秒")
    
    # 保存结果
    result_file = os.path.join(output_dir, 'evaluation_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("指纹匹配算法评估结果\n")
        f.write("="*60 + "\n\n")
        f.write(f"数据集目录: {args.dataset_dir}\n")
        f.write(f"mnt文件目录: {args.mnt_dir}\n")
        f.write(f"模板目录: {template_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"图像格式: {args.image_format if args.image_format else '自动检测'}\n")
        f.write(f"正样本数量: {len(all_genuine_scores)}\n")
        f.write(f"负样本数量: {len(all_impostor_scores)}\n\n")
        
        f.write("匹配器参数:\n")
        f.write(f"  k_neighbors: {args.k_neighbors}\n")
        f.write(f"  template_radius: {args.template_radius}\n")
        f.write(f"  distance_tolerance: {args.distance_tolerance}\n")
        f.write(f"  angle_tolerance: {args.angle_tolerance}\n")
        f.write(f"  orientation_tolerance: {args.orientation_tolerance}\n")
        f.write(f"  max_distance: {args.max_distance}\n")
        f.write(f"  match_threshold: {args.match_threshold}\n\n")
        
        f.write("不同阈值下的FAR和FRR:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'阈值':<10} {'FAR':<15} {'FRR':<15} {'准确率':<15}\n")
        f.write("-"*60 + "\n")
        for threshold, far, frr in results:
            accuracy = (1 - far - frr) / 2 + 0.5
            f.write(f"{threshold:<10.2f} {far:<15.4f} {frr:<15.4f} {accuracy:<15.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write(f"EER (等错误率): {eer:.4f} ({eer*100:.2f}%)\n")
        f.write(f"EER对应的匹配分数阈值: {eer_threshold:.4f}\n\n")
        
        f.write("时间统计:\n")
        f.write("-"*60 + "\n")
        f.write(f"模板构建时间: {template_build_time:.3f} 秒\n")
        f.write(f"匹配计算时间: {matching_time:.3f} 秒\n")
        f.write(f"  总匹配次数: {match_stats['total_matches']} (正样本: {match_stats['genuine_matches']}, 负样本: {match_stats['impostor_matches']})\n")
        f.write(f"  平均每次匹配时间: {match_stats['avg_match_time']*1000:.3f} 毫秒\n")
        f.write(f"  平均正样本匹配时间: {match_stats['avg_genuine_time']*1000:.3f} 毫秒\n")
        f.write(f"  平均负样本匹配时间: {match_stats['avg_impostor_time']*1000:.3f} 毫秒\n")
        f.write(f"评估曲线图绘制时间: {plot_time:.3f} 秒\n")
        f.write(f"可视化生成时间: {visualization_time:.3f} 秒\n")
        total_time = template_build_time + matching_time + plot_time + visualization_time
        f.write(f"总时间: {total_time:.3f} 秒\n")
    
    print(f"\n所有结果文件已保存到目录: {output_dir}")
    print(f"  - 结果文本文件: {os.path.basename(result_file)}")
    print(f"  - ROC曲线图: {os.path.basename(roc_path)}")
    print(f"  - 分数分布图: {os.path.basename(dist_path)}")
    print(f"  - FAR-FRR 曲线图: {os.path.basename(far_frr_path)}")
    print(f"  - DET 曲线图: {os.path.basename(det_path)}")
    print(f"  - Precision-Recall 曲线图: {os.path.basename(pr_path)}")
    
    # 输出可视化文件信息
    print("\n匹配可视化图:")
    success_files = visualization_outputs.get('success', [])
    if success_files:
        print(f"  - 成功匹配示例 ({len(success_files)} 张):")
        for file_path in success_files:
            print(f"      {os.path.relpath(file_path, output_dir)}")
    else:
        print("  - 成功匹配示例: 未生成（可能缺少有效样本）")
    
    false_negative_files = visualization_outputs.get('false_negative', [])
    if false_negative_files:
        print(f"  - 假阴性示例 ({len(false_negative_files)} 张，本该成功却失败):")
        for file_path in false_negative_files:
            print(f"      {os.path.relpath(file_path, output_dir)}")
    else:
        print("  - 假阴性示例: 未生成（可能不存在此类样本）")
    
    false_positive_files = visualization_outputs.get('false_positive', [])
    if false_positive_files:
        print(f"  - 假阳性示例 ({len(false_positive_files)} 张，本该失败却成功):")
        for file_path in false_positive_files:
            print(f"      {os.path.relpath(file_path, output_dir)}")
    else:
        print("  - 假阳性示例: 未生成（可能不存在此类样本）")


if __name__ == "__main__":
    main()

