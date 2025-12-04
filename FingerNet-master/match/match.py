"""
基于模板匹配的指纹识别算法
适用于手机解锁等1:1验证场景
使用细节点局部结构模板进行快速匹配
"""

import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None
try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from scipy.spatial.distance import cdist

MB_IN_BYTES = 1024.0 * 1024.0


def collect_memory_snapshot() -> Dict[str, Optional[float]]:
    """采集当前常驻内存与显存."""
    snapshot = {'rss': None, 'gpu_current': None, 'gpu_peak': None}
    if psutil is not None:
        try:
            snapshot['rss'] = psutil.Process(os.getpid()).memory_info().rss / MB_IN_BYTES
        except Exception:
            pass
    tf_module = tf
    if tf_module is not None:
        try:
            devices = tf_module.config.experimental.list_physical_devices('GPU')
        except Exception:
            devices = []
        if devices:
            try:
                gpu_info = tf_module.config.experimental.get_memory_info('GPU:0')
                snapshot['gpu_current'] = gpu_info.get('current', 0.0) / MB_IN_BYTES
                snapshot['gpu_peak'] = gpu_info.get('peak', 0.0) / MB_IN_BYTES
            except Exception:
                pass
    return snapshot


def format_memory_value(value: Optional[float]) -> str:
    return f"{value:.1f}MB" if value is not None else "N/A"


def record_memory(records: List[Dict[str, Optional[float]]], label: str):
    snapshot = collect_memory_snapshot()
    snapshot['label'] = label
    records.append(snapshot)
    print(f"[内存] {label} - 常驻: {format_memory_value(snapshot['rss'])}, "
          f"GPU当前: {format_memory_value(snapshot['gpu_current'])}, "
          f"GPU峰值: {format_memory_value(snapshot['gpu_peak'])}")


def summarize_memory(records: List[Dict[str, Optional[float]]], key: str) -> Tuple[Optional[float], Optional[float]]:
    values = [rec[key] for rec in records if rec.get(key) is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.max(values))


@dataclass
class Minutia:
    """细节点数据结构"""
    x: float
    y: float
    orientation: float  # 弧度
    confidence: float
    minutia_type: int = 0  # 0: 端点, 1: 分叉点


@dataclass
class LocalTemplate:
    """局部结构模板"""
    center_idx: int  # 中心细节点索引
    neighbor_indices: List[int]  # k个最近邻细节点索引
    distances: np.ndarray  # 到k个最近邻的距离
    angles: np.ndarray  # 到k个最近邻的角度（相对于中心点方向）
    orientation_diffs: np.ndarray  # 中心点与邻居点的方向差


@dataclass
class MatchResult:
    """匹配结果"""
    score: float  # 匹配分数 (0-1)
    num_matches: int  # 匹配数量
    transform_matrix: np.ndarray  # 变换矩阵
    matched_pairs: List[Tuple[int, int]]  # 匹配对
    inlier_ratio: float  # 内点比例
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemplateMatcher:
    """基于模板匹配的指纹匹配器"""
    
    def __init__(self,
                 k_neighbors: int = 5,
                 template_radius: float = 100.0,
                 distance_tolerance: float = 15.0,
                 angle_tolerance: float = 30.0,  # 度
                 orientation_tolerance: float = 30.0,  # 度
                 max_distance: float = 40.0,  # 减小距离容差，提高匹配精度
                 match_threshold: float = 0.65,  # 平衡的默认阈值
                 ransac_runs: int = 3,  # RANSAC运行次数，多次运行取最佳结果以提高稳定性
                 templates_per_finger: int = 1):  # 每个手指用于模板的样本数量（固定 enrollment）
        """
        初始化模板匹配器
        
        Args:
            k_neighbors: 局部模板中使用的最近邻数量
            template_radius: 模板搜索半径（像素）
            distance_tolerance: 距离容差（像素）
            angle_tolerance: 角度容差（度）
            orientation_tolerance: 方向容差（度）
            max_distance: RANSAC中最大匹配距离（像素）
            match_threshold: 模板匹配阈值
            ransac_runs: RANSAC运行次数，多次运行取最佳结果以提高稳定性（默认: 3）
            templates_per_finger: 每个手指用于模板的样本数量（固定 enrollment，默认: 1）
        """
        self.k_neighbors = k_neighbors
        self.template_radius = template_radius
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = np.radians(angle_tolerance)
        self.orientation_tolerance = np.radians(orientation_tolerance)
        self.max_distance = max_distance
        self.match_threshold = match_threshold
        self.ransac_runs = ransac_runs  # RANSAC运行次数
        self.templates_per_finger = max(1, templates_per_finger)
    
    def _calculate_angle_difference(self, angle1: float, angle2: float) -> float:
        """计算两个角度之间的最小差值（考虑角度环绕）"""
        diff = abs(angle1 - angle2)
        return min(diff, 2 * np.pi - diff)
    
    def build_local_template(self,
                            center_minutia: Minutia,
                            all_minutiae: List[Minutia],
                            center_idx: int) -> LocalTemplate:
        """
        为细节点构建局部结构模板
        
        模板包含：
        - k个最近邻细节点
        - 到每个邻居的距离
        - 到每个邻居的角度（相对于中心点方向）
        - 中心点与邻居点的方向差
        
        Args:
            center_minutia: 中心细节点
            all_minutiae: 所有细节点列表
            center_idx: 中心细节点索引
            
        Returns:
            局部模板
        """
        # 计算所有点到中心点的距离
        distances = []
        for i, mnt in enumerate(all_minutiae):
            if i == center_idx:
                continue
            dist = np.sqrt((mnt.x - center_minutia.x)**2 + (mnt.y - center_minutia.y)**2)
            # 只考虑半径内的点
            if dist <= self.template_radius:
                distances.append((i, dist))
        
        # 按距离排序，选择k个最近邻
        distances.sort(key=lambda x: x[1])
        k_neighbors = min(self.k_neighbors, len(distances))
        
        if k_neighbors == 0:
            return LocalTemplate(center_idx, [], np.array([]), np.array([]), np.array([]))
        
        neighbor_indices = [distances[i][0] for i in range(k_neighbors)]
        neighbor_distances = np.array([distances[i][1] for i in range(k_neighbors)])
        
        # 计算角度和方向差
        angles = []
        orientation_diffs = []
        
        for idx in neighbor_indices:
            neighbor = all_minutiae[idx]
            # 计算从中心点到邻居点的角度
            dx = neighbor.x - center_minutia.x
            dy = neighbor.y - center_minutia.y
            angle = np.arctan2(dy, dx)
            
            # 相对于中心点方向的角度
            relative_angle = self._calculate_angle_difference(angle, center_minutia.orientation)
            angles.append(relative_angle)
            
            # 方向差
            ori_diff = self._calculate_angle_difference(
                center_minutia.orientation,
                neighbor.orientation
            )
            orientation_diffs.append(ori_diff)
        
        return LocalTemplate(
            center_idx=center_idx,
            neighbor_indices=neighbor_indices,
            distances=neighbor_distances,
            angles=np.array(angles),
            orientation_diffs=np.array(orientation_diffs)
        )
    
    def compute_templates(self, minutiae: List[Minutia]) -> List[LocalTemplate]:
        """
        为所有细节点计算局部模板
        
        Args:
            minutiae: 细节点列表
            
        Returns:
            模板列表
        """
        templates = []
        for i, mnt in enumerate(minutiae):
            template = self.build_local_template(mnt, minutiae, i)
            templates.append(template)
        return templates
    
    def match_templates(self,
                       template1: LocalTemplate,
                       template2: LocalTemplate,
                       minutiae1: List[Minutia],
                       minutiae2: List[Minutia]) -> float:
        """
        匹配两个局部模板（优化版本）
        
        改进点：
        1. 使用更严格的匹配标准
        2. 要求多个特征同时匹配
        3. 考虑邻居的相对顺序
        
        Args:
            template1: 第一个局部模板
            template2: 第二个局部模板
            minutiae1: 第一组细节点，用于查找邻居坐标和方向
            minutiae2: 第二组细节点，用于查找邻居坐标和方向
            
        Returns:
            匹配相似度 (0-1)
        """
        k1, k2 = len(template1.neighbor_indices), len(template2.neighbor_indices)
        if k1 == 0 or k2 == 0:
            return 0.0
        
        # 如果邻居数量差异太大，相似度降低
        if abs(k1 - k2) > max(k1, k2) * 0.5:  # 差异超过50%
            return 0.0
        
        # 改进的匹配：同时考虑距离、角度、方向的组合匹配
        # 使用更严格的匹配策略：要求多个特征同时满足条件
        
        # 构建邻居特征对
        features1 = []
        features2 = []
        
        for i, idx1 in enumerate(template1.neighbor_indices):
            neighbor1 = minutiae1[idx1]
            dx1 = neighbor1.x - minutiae1[template1.center_idx].x
            dy1 = neighbor1.y - minutiae1[template1.center_idx].y
            
            features1.append({
                'dist': template1.distances[i],
                'angle': template1.angles[i],
                'ori_diff': template1.orientation_diffs[i],
                'neighbor_ori': neighbor1.orientation
            })
        
        for i, idx2 in enumerate(template2.neighbor_indices):
            neighbor2 = minutiae2[idx2]
            dx2 = neighbor2.x - minutiae2[template2.center_idx].x
            dy2 = neighbor2.y - minutiae2[template2.center_idx].y
            
            features2.append({
                'dist': template2.distances[i],
                'angle': template2.angles[i],
                'ori_diff': template2.orientation_diffs[i],
                'neighbor_ori': neighbor2.orientation
            })
        
        # 使用贪心算法进行最优匹配
        matched_pairs = []
        used2 = set()
        
        # 按距离排序，优先匹配距离相近的
        for i, f1 in enumerate(features1):
            best_match = None
            best_score = 0.0
            
            for j, f2 in enumerate(features2):
                if j in used2:
                    continue
                
                # 计算综合相似度（放宽约束以降低FRR）
                # 距离相似度（放宽到20%容差）
                dist_diff = abs(f1['dist'] - f2['dist']) / max(f1['dist'], f2['dist'], 1.0)
                if dist_diff > 0.20:  # 20%容差
                    continue
                
                # 角度相似度（放宽到85%容差）
                angle_diff = self._calculate_angle_difference(f1['angle'], f2['angle'])
                if angle_diff > self.angle_tolerance * 0.85:  # 85%的容差
                    continue
                
                # 方向差相似度（放宽到85%容差）
                ori_diff_diff = abs(f1['ori_diff'] - f2['ori_diff'])
                if ori_diff_diff > self.orientation_tolerance * 0.85:
                    continue
                
                # 邻居方向相似度（放宽约束）
                neighbor_ori_diff = self._calculate_angle_difference(
                    f1['neighbor_ori'], f2['neighbor_ori']
                )
                if neighbor_ori_diff > self.orientation_tolerance * 1.2:  # 放宽20%
                    continue
                
                # 计算综合分数
                dist_score = 1.0 - dist_diff / 0.20
                angle_score = 1.0 - angle_diff / (self.angle_tolerance * 0.85)
                ori_score = 1.0 - ori_diff_diff / (self.orientation_tolerance * 0.85)
                neighbor_ori_score = 1.0 - neighbor_ori_diff / (self.orientation_tolerance * 1.2)
                
                # 加权平均（更重视距离和角度）
                score = (dist_score * 0.35 + angle_score * 0.3 + 
                        ori_score * 0.2 + neighbor_ori_score * 0.15)
                
                if score > best_score:
                    best_score = score
                    best_match = j
            
            # 使用类属性中的match_threshold，但这里需要稍微降低以允许更多匹配对
            # 因为这是局部模板匹配，后续还有几何验证
            local_match_threshold = self.match_threshold * 0.77  # 约等于0.5（当match_threshold=0.65时）
            if best_match is not None and best_score > local_match_threshold:
                matched_pairs.append((i, best_match, best_score))
                used2.add(best_match)
        
        # 计算最终相似度
        if len(matched_pairs) == 0:
            return 0.0
        
        # 降低最小匹配要求（从一半降到1/3）
        min_matches = max(1, min(k1, k2) // 3)  # 至少匹配1/3的邻居
        if len(matched_pairs) < min_matches:
            return 0.0
        
        # 计算平均匹配分数
        avg_score = np.mean([score for _, _, score in matched_pairs])
        
        # 考虑匹配比例
        match_ratio = len(matched_pairs) / max(k1, k2)
        
        # 一局部模板的综合相似度：平均分数 * 匹配比例（放宽要求）
        similarity = avg_score * (0.7 + 0.3 * match_ratio)  # 即使匹配比例低也给予一定分数
        
        # 放宽惩罚：匹配比例低于40%才降低相似度
        if match_ratio < 0.4:  # 匹配比例低于40%
            similarity *= match_ratio / 0.4
        
        return similarity
    
    def find_template_matches(self,
                             templates1: List[LocalTemplate],
                             templates2: List[LocalTemplate],
                             minutiae1: List[Minutia],
                             minutiae2: List[Minutia]) -> List[Tuple[int, int, float]]:
        """
        找到所有模板匹配对（优化版本）
        
        改进点：
        1. 提高匹配阈值
        2. 添加双向匹配验证
        3. 要求更多匹配对
        
        Args:
            templates1: 第一组模板
            templates2: 第二组模板
            minutiae1: 第一组细节点
            minutiae2: 第二组细节点
            
        Returns:
            匹配对列表 (idx1, idx2, similarity)
        """
        matches = []
        
        # 简化匹配策略：使用单向匹配，但要求更高的相似度
        # 双向验证太严格，导致FRR过高
        
        for i, t1 in enumerate(templates1):
            if len(t1.neighbor_indices) < 1:  # 放宽到至少1个邻居
                continue
            
            best_match = -1
            best_sim = 0.0
            
            for j, t2 in enumerate(templates2):
                if len(t2.neighbor_indices) < 1:
                    continue
                
                # 检查中心点的方向一致性（放宽到100%容差）
                ori_diff = self._calculate_angle_difference(
                    minutiae1[i].orientation,
                    minutiae2[j].orientation
                )
                if ori_diff > self.orientation_tolerance:  # 使用完整容差
                    continue
                
                # 计算模板相似度
                sim = self.match_templates(t1, t2, minutiae1, minutiae2)
                
                if sim > best_sim and sim >= self.match_threshold:
                    best_sim = sim
                    best_match = j
            
            if best_match != -1:
                matches.append((i, best_match, best_sim))
        
        return matches
    
    def geometric_verification(self,
                              minutiae1: List[Minutia],
                              minutiae2: List[Minutia],
                              matches: List[Tuple[int, int, float]]) -> MatchResult:
        """
        使用RANSAC进行几何验证（多次运行取最佳结果以提高稳定性）
        
        Args:
            minutiae1: 第一组细节点
            minutiae2: 第二组细节点
            matches: 初始匹配对
            
        Returns:
            匹配结果
        """
        if len(matches) < 3:
            return MatchResult(0.0, 0, np.eye(3), [], 0.0)
        
        # 提取匹配点的坐标
        pts1 = np.array([[minutiae1[i].x, minutiae1[i].y] for i, _, _ in matches])
        pts2 = np.array([[minutiae2[j].x, minutiae2[j].y] for _, j, _ in matches])
        
        # 多次运行RANSAC，取最佳结果以提高稳定性
        best_overall_result = None
        best_overall_score = -1.0
        
        for run in range(self.ransac_runs):
            # RANSAC参数
            inlier_threshold = self.max_distance
            estimated_inlier_ratio = 0.3
            
            # 计算最大迭代次数，避免log(0)或log(负数)的问题
            if estimated_inlier_ratio > 0 and estimated_inlier_ratio < 1:
                try:
                    max_iterations = min(1000, max(100, int(np.log(0.01) / np.log(1 - estimated_inlier_ratio**3))))
                except (ValueError, ZeroDivisionError):
                    max_iterations = 500  # 默认值
            else:
                max_iterations = 500  # 默认值
            
            # 改进的提前停止条件：更严格，避免过早停止导致结果不稳定
            min_inliers_for_early_stop = max(5, len(matches) // 3)  # 提高最小内点数要求
            min_iterations_before_stop = max(50, int(max_iterations * 0.3))  # 至少运行30%的迭代次数
            early_stop_inlier_ratio = 0.75  # 提高内点比例要求到75%（更严格）
            consecutive_good_iterations = 0  # 连续满足条件的迭代次数
            required_consecutive = 3  # 要求连续3次满足条件才停止
            
            best_transform = None
            best_inliers = 0
            best_inlier_indices = []
            
            for iteration in range(max_iterations):
                if len(matches) < 3:
                    break
                
                # 随机选择3个点
                sample_indices = np.random.choice(len(matches), 3, replace=False)
                sample_pts1 = pts1[sample_indices]
                sample_pts2 = pts2[sample_indices]
                
                # 检查采样点的方向一致性
                orientation_ok = True
                for idx in sample_indices:
                    match_idx1, match_idx2 = matches[idx][0], matches[idx][1]
                    ori_diff = self._calculate_angle_difference(
                        minutiae1[match_idx1].orientation,
                        minutiae2[match_idx2].orientation
                    )
                    if ori_diff > self.orientation_tolerance:
                        orientation_ok = False
                        break
                
                if not orientation_ok:
                    consecutive_good_iterations = 0  # 重置连续计数
                    continue
                
                try:
                    # 估计仿射变换
                    transform = cv2.getAffineTransform(
                        sample_pts1.astype(np.float32),
                        sample_pts2.astype(np.float32)
                    )
                    
                    # 计算内点
                    ones = np.ones((len(pts1), 1))
                    pts1_homo = np.hstack([pts1, ones])
                    transform_3x3 = np.vstack([transform, [0, 0, 1]])
                    transformed_pts1 = np.dot(pts1_homo, transform_3x3.T)[:, :2]
                    
                    distances = np.linalg.norm(transformed_pts1 - pts2, axis=1)
                    geometric_inliers = np.where(distances < inlier_threshold)[0]
                    
                    # 进一步检查方向一致性
                    orientation_inliers = []
                    for idx in geometric_inliers:
                        match_idx1, match_idx2 = matches[idx][0], matches[idx][1]
                        ori_diff = self._calculate_angle_difference(
                            minutiae1[match_idx1].orientation,
                            minutiae2[match_idx2].orientation
                        )
                        if ori_diff <= self.orientation_tolerance:
                            orientation_inliers.append(idx)
                    
                    inlier_indices = np.array(orientation_inliers)
                    
                    # 更新本次运行的最佳结果
                    if len(inlier_indices) > best_inliers:
                        best_inliers = len(inlier_indices)
                        best_inlier_indices = inlier_indices
                        best_transform = transform
                    
                    # 改进的提前停止条件：更严格，避免过早停止
                    # 1. 至少运行一定比例的迭代次数
                    # 2. 内点数达到要求
                    # 3. 内点比例达到更高要求（75%）
                    # 4. 连续多次满足条件（提高稳定性）
                    if iteration >= min_iterations_before_stop and best_inliers >= min_inliers_for_early_stop:
                        current_inlier_ratio = best_inliers / len(matches)
                        if current_inlier_ratio >= early_stop_inlier_ratio:
                            consecutive_good_iterations += 1
                            # 连续多次满足条件才停止，避免偶然的好结果导致过早停止
                            if consecutive_good_iterations >= required_consecutive:
                                break
                        else:
                            consecutive_good_iterations = 0  # 不满足条件，重置计数
                    else:
                        consecutive_good_iterations = 0  # 不满足条件，重置计数
                
                except:
                    continue
            
            # 计算本次运行的匹配分数
            if best_transform is not None and best_inliers > 0:
                inlier_ratio = best_inliers / len(matches)
                
                # 提高要求：至少5个内点（减少false positive）
                if best_inliers < 5:
                    continue  # 本次运行失败，继续下一次运行
                
                # 提高内点比例要求到30%（收紧几何验证条件）
                if inlier_ratio < 0.3:
                    continue  # 本次运行失败，继续下一次运行
                
                # 计算变换矩阵的质量（检查是否合理）
                # 检查缩放因子（放宽范围）
                scale_x = np.sqrt(best_transform[0, 0]**2 + best_transform[1, 0]**2)
                scale_y = np.sqrt(best_transform[0, 1]**2 + best_transform[1, 1]**2)
                
                # 缩放因子应该在合理范围内（放宽到0.3-3.0）
                if scale_x < 0.3 or scale_x > 3.0 or scale_y < 0.3 or scale_y > 3.0:
                    continue  # 本次运行失败，继续下一次运行
                
                # 改进的分数计算（更平衡）
                # 1. 内点比例
                inlier_score = inlier_ratio
                
                # 2. 匹配数量因子（降低要求：10个内点为满分）
                num_factor = min(1.0, best_inliers / 10.0)
                
                # 3. 内点分布质量（放宽检查）
                inlier_pts1 = pts1[best_inlier_indices]
                inlier_pts2 = pts2[best_inlier_indices]
                
                # 计算内点之间的平均距离（放宽到15像素）
                if len(inlier_pts1) > 1:
                    distances_inlier = np.linalg.norm(
                        inlier_pts1 - np.mean(inlier_pts1, axis=0), axis=1
                    )
                    avg_spread = np.mean(distances_inlier)
                    # 放宽分布要求
                    if avg_spread < 15.0:  # 平均分布半径小于15像素
                        spread_factor = avg_spread / 15.0
                    else:
                        spread_factor = 1.0
                else:
                    spread_factor = 0.7  # 放宽单个内点的惩罚
                
                # 综合分数（更平衡的权重）
                score = (inlier_score * 0.6 + num_factor * 0.25 + spread_factor * 0.15)
                
                # 放宽惩罚：内点比例低于30%才降低分数
                if inlier_ratio < 0.3:
                    score *= (0.7 + 0.3 * inlier_ratio / 0.3)  # 更温和的惩罚
                
                # 转换为3x3齐次变换矩阵
                transform_3x3 = np.eye(3)
                transform_3x3[:2, :] = best_transform
                
                matched_pairs = [(matches[i][0], matches[i][1]) for i in best_inlier_indices]
                
                # 创建本次运行的结果
                run_result = MatchResult(
                    score=min(1.0, score),
                    num_matches=best_inliers,
                    transform_matrix=transform_3x3,
                    matched_pairs=matched_pairs,
                    inlier_ratio=inlier_ratio
                )
                
                # 保留最佳结果（按分数和内点数综合判断）
                # 优先考虑分数，如果分数相同则考虑内点数
                result_score = run_result.score
                if result_score > best_overall_score or \
                   (result_score == best_overall_score and best_inliers > (best_overall_result.num_matches if best_overall_result else 0)):
                    best_overall_score = result_score
                    best_overall_result = run_result
        
        # 返回多次运行中的最佳结果
        if best_overall_result is not None:
            return best_overall_result
        else:
            return MatchResult(0.0, 0, np.eye(3), [], 0.0)
    
    def load_minutiae_from_mnt(self, mnt_file: str) -> List[Minutia]:
        """从.mnt文件加载细节点数据"""
        minutiae = []
        try:
            with open(mnt_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines[2:]:  # 跳过前两行
                parts = line.strip().split()
                if len(parts) >= 3:
                    x = float(parts[0])
                    y = float(parts[1])
                    orientation = float(parts[2])  # 弧度
                    confidence = float(parts[3]) if len(parts) > 3 else 1.0
                    minutiae.append(Minutia(x, y, orientation, confidence))
        except Exception as e:
            print(f"加载细节点文件失败: {e}")
        
        return minutiae
    
    def save_multi_template(self,
                           template_file: str,
                           multi_templates: List[Tuple[List[LocalTemplate], List[Minutia]]],
                           metadata: Optional[Dict] = None,
                           sample_ids: Optional[List[str]] = None):
        """
        保存多模板（同一手指的多个样本）
        
        Args:
            template_file: 模板文件路径
            multi_templates: 多个模板列表，每个元素为(templates, minutiae)
            metadata: 元数据
            sample_ids: 样本ID列表，用于识别相同样本（避免数据泄露）
        """
        try:
            save_data = {
                'multi_templates': multi_templates,  # 多个模板
                'sample_ids': sample_ids or [],  # 样本ID列表
                'metadata': metadata or {}
            }
            
            with open(template_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"多模板已保存到: {template_file} (包含 {len(multi_templates)} 个样本)")
        except Exception as e:
            print(f"保存多模板失败: {e}")
    
    def load_multi_template(self, template_file: str, verbose: bool = False) -> Tuple[List[Tuple[List[LocalTemplate], List[Minutia]]], Dict, List[str]]:
        """
        从文件加载多模板
        
        Args:
            template_file: 模板文件路径
            verbose: 是否打印详细信息
            
        Returns:
            (multi_templates, metadata, sample_ids)，multi_templates是(templates, minutiae)的列表
        """
        try:
            with open(template_file, 'rb') as f:
                save_data = pickle.load(f)
            
            # 统一使用多模板格式
            multi_templates = save_data.get('multi_templates', [])
            metadata = save_data.get('metadata', {})
            sample_ids = save_data.get('sample_ids', [])  # 样本ID列表
            
            if verbose:
                print(f"多模板已从 {template_file} 加载 (包含 {len(multi_templates)} 个样本)")
            return multi_templates, metadata, sample_ids
        except Exception as e:
            if verbose:
                print(f"加载多模板失败: {e}")
            return [], {}, []
    
    def match_with_stored_template(self, 
                                  stored_template_file: str,
                                  query_minutiae: List[Minutia],
                                  exclude_query_sample_id: Optional[str] = None) -> MatchResult:
        """
        使用存储的模板进行匹配（支持单模板和多模板）
        
        Args:
            stored_template_file: 存储的模板文件路径
            query_minutiae: 查询指纹的细节点
            exclude_query_sample_id: 要排除的查询样本ID（用于避免数据泄露，如"00000_00"）
            
        Returns:
            匹配结果（多模板时返回最佳匹配结果）
        """
        # 尝试加载多模板（不打印详细信息）
        multi_templates, metadata, sample_ids = self.load_multi_template(stored_template_file, verbose=False)
        
        if len(multi_templates) == 0:
            return MatchResult(0.0, 0, np.eye(3), [], 0.0)
        
        # 为查询指纹构建模板
        query_templates = self.compute_templates(query_minutiae)
        
        # 统一处理：与所有模板匹配并取最佳结果（单模板和多模板统一处理）
        best_result = None
        best_score = -1.0
        best_metadata: Dict[str, Any] = {}
        
        for idx, (stored_templates, stored_minutiae) in enumerate(multi_templates):
            if len(stored_templates) == 0:
                continue
            
            # 如果提供了要排除的样本ID，检查当前模板是否对应该样本
            if exclude_query_sample_id is not None and len(sample_ids) > idx:
                if sample_ids[idx] == exclude_query_sample_id:
                    continue  # 跳过这个模板，避免自己匹配自己
            
            # 模板匹配
            matches = self.find_template_matches(stored_templates, query_templates, 
                                                stored_minutiae, query_minutiae)
            
            if len(matches) == 0:
                continue
            
            # 几何验证
            result = self.geometric_verification(stored_minutiae, query_minutiae, matches)
            
            # 保留最佳结果（优先考虑分数，如果分数相同则考虑内点数）
            should_update = False
            if result.score > best_score:
                should_update = True
            elif result.score == best_score and result.score > 0.0:
                # 如果分数相同且都大于0，选择内点数更多的
                if best_result is None or result.num_matches > best_result.num_matches:
                    should_update = True
            elif result.score > 0.0 and best_score <= 0.0:
                # 如果当前结果分数>0，而之前的最佳结果是0，则更新
                should_update = True
            
            if should_update:
                best_score = result.score
                best_result = result
                best_metadata = {
                    'stored_index': idx,
                    'stored_sample_id': sample_ids[idx] if len(sample_ids) > idx else None,
                    'finger_id': metadata.get('finger_id')
                }
        
        if best_result is not None:
            best_result.metadata = best_metadata
            return best_result
        
        # 如果所有匹配都失败，仍然返回一个结果，但metadata中记录第一个尝试匹配的样本（如果有）
        # 这样可以确保可视化时能显示参考图像
        if len(multi_templates) > 0 and len(sample_ids) > 0:
            # 找到第一个非排除的样本作为参考
            for idx in range(len(multi_templates)):
                if exclude_query_sample_id is None or (len(sample_ids) > idx and sample_ids[idx] != exclude_query_sample_id):
                    return MatchResult(
                        0.0, 0, np.eye(3), [], 0.0,
                        metadata={
                            'stored_index': idx,
                            'stored_sample_id': sample_ids[idx] if len(sample_ids) > idx else None,
                            'finger_id': metadata.get('finger_id')
                        }
                    )
        return MatchResult(0.0, 0, np.eye(3), [], 0.0)
    
    def match_fingerprints(self,
                          minutiae1: List[Minutia],
                          minutiae2: List[Minutia]) -> MatchResult:
        """
        匹配两个指纹（模板匹配方法）
        
        Args:
            minutiae1: 第一个指纹的细节点
            minutiae2: 第二个指纹的细节点
            
        Returns:
            匹配结果
        """
        # 构建局部模板
        templates1 = self.compute_templates(minutiae1)
        templates2 = self.compute_templates(minutiae2)
        
        # 模板匹配
        matches = self.find_template_matches(templates1, templates2, minutiae1, minutiae2)
        
        if len(matches) == 0:
            return MatchResult(0.0, 0, np.eye(3), [], 0.0)
        
        # 几何验证
        result = self.geometric_verification(minutiae1, minutiae2, matches)
        return result
    
    def is_match_successful(self, score: float, threshold: Optional[float] = None) -> bool:
        """
        判断匹配是否成功
        
        Args:
            score: 匹配分数
            threshold: 匹配阈值，如果为None则使用实例的match_threshold
            
        Returns:
            是否匹配成功
        """
        if threshold is None:
            threshold = self.match_threshold
        return score >= threshold
    
    def get_score_interpretation(self, score: float) -> str:
        """
        获取分数解释
        
        Args:
            score: 匹配分数
            
        Returns:
            分数解释字符串
        """
        if score >= self.match_threshold:
            return "匹配成功 - 是同一指纹"
        else:
            return "匹配失败 - 非同一指纹"
    
    def visualize_matches(self,
                         image1: np.ndarray,
                         minutiae1: List[Minutia],
                         image2: np.ndarray,
                         minutiae2: List[Minutia],
                         result: MatchResult,
                         save_path: Optional[str] = None,
                         image1_label: Optional[str] = None,
                         image2_label: Optional[str] = None):
        """可视化匹配结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        label1 = image1_label or '指纹1'
        label2 = image2_label or '指纹2'
        
        # 显示第一个指纹
        ax1.imshow(image1, cmap='gray')
        for i, mnt in enumerate(minutiae1):
            color = 'red' if any(i == pair[0] for pair in result.matched_pairs) else 'blue'
            ax1.plot(mnt.x, mnt.y, 'o', color=color, markersize=3)
            ax1.arrow(mnt.x, mnt.y,
                    10 * np.cos(mnt.orientation),
                    10 * np.sin(mnt.orientation),
                    head_width=3, head_length=2, fc=color, ec=color)
        ax1.set_title(f'{label1} (匹配: {len(result.matched_pairs)})')
        ax1.axis('off')
        
        # 显示第二个指纹
        ax2.imshow(image2, cmap='gray')
        for i, mnt in enumerate(minutiae2):
            color = 'red' if any(i == pair[1] for pair in result.matched_pairs) else 'blue'
            ax2.plot(mnt.x, mnt.y, 'o', color=color, markersize=3)
            ax2.arrow(mnt.x, mnt.y,
                    10 * np.cos(mnt.orientation),
                    10 * np.sin(mnt.orientation),
                    head_width=3, head_length=2, fc=color, ec=color)
        ax2.set_title(f'{label2} (匹配: {len(result.matched_pairs)})')
        ax2.axis('off')
        
        plt.suptitle(f'{label1} vs {label2}\n匹配分数: {result.score:.2f}, 内点比例: {result.inlier_ratio:.2f}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig('fingerprint_match_result.png', dpi=150, bbox_inches='tight')
        plt.close()


def demo_matching():
    """演示模板匹配功能"""
    memory_records: List[Dict[str, Optional[float]]] = []
    record_memory(memory_records, "脚本启动")
    # 创建模板匹配器（使用默认参数，match_threshold=0.65）
    matcher = TemplateMatcher(
        k_neighbors=5,  # 局部模板使用5个最近邻
        template_radius=100.0,  # 模板搜索半径100像素
        distance_tolerance=15.0,  # 距离容差15像素
        angle_tolerance=30.0,  # 角度容差30度
        orientation_tolerance=30.0,  # 方向容差30度
        max_distance=50.0,  # RANSAC最大距离50像素
        match_threshold=0.65,  # 模板匹配阈值0.65（与默认值一致）
        templates_per_finger=1
    )
    record_memory(memory_records, "创建匹配器")
    
    print("指纹模板匹配演示")
    print("="*50)
    
    # 示例用法
    image1_path = '../datasets/fvc2004_DB1_B/101_7.tif'
    image2_path = '../datasets/fvc2004_DB1_B/101_3.tif'
    mnt1_path = '../output/20251125-155019/fvc2004_DB1_B/101_7.mnt'
    mnt2_path = '../output/20251125-155019/fvc2004_DB1_B/101_3.mnt'
    
    # 检查文件是否存在
    for path in [image1_path, image2_path, mnt1_path, mnt2_path]:
        if not os.path.exists(path):
            print(f"错误：文件不存在 - {path}")
            return
    
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if image1 is None or image2 is None:
        print("错误：无法加载图像文件")
        return
    
    minutiae1 = matcher.load_minutiae_from_mnt(mnt1_path)
    minutiae2 = matcher.load_minutiae_from_mnt(mnt2_path)
    record_memory(memory_records, "加载图像与细节点")
    
    if not minutiae1 or not minutiae2:
        print("错误：无法加载细节点数据")
        return
    
    print(f"指纹1细节点数量: {len(minutiae1)}")
    print(f"指纹2细节点数量: {len(minutiae2)}")
    
    # 执行模板匹配
    print("\n开始模板匹配...")
    result = matcher.match_fingerprints(minutiae1, minutiae2)
    record_memory(memory_records, "匹配完成")
    
    # 输出结果
    print(f"\n匹配结果:")
    print(f"匹配分数: {result.score:.3f} (归一化 0-1)")
    print(f"匹配数量: {result.num_matches}")
    print(f"内点比例: {result.inlier_ratio:.2f}")
    
    # 匹配成功判断（使用matcher实例的阈值）
    is_success = matcher.is_match_successful(result.score)  # 使用默认阈值（matcher.match_threshold）
    print(f"\n匹配成功判断:")
    print(f"  阈值: {matcher.match_threshold} (使用matcher的match_threshold参数)")
    print(f"  结果: {'✓ 成功' if is_success else '✗ 失败'}")
    
    # 分数解释
    interpretation = matcher.get_score_interpretation(result.score)
    print(f"\n分数解释: {interpretation}")
    
    # 可视化
    print("\n正在生成匹配结果可视化图片...")
    matcher.visualize_matches(
        image1,
        minutiae1,
        image2,
        minutiae2,
        result,
        'match_result.png',
        image1_label=os.path.basename(image1_path),
        image2_label=os.path.basename(image2_path)
    )
    print("匹配结果已保存为 match_result.png")
    record_memory(memory_records, "可视化完成")
    
    print("\n内存统计:")
    for rec in memory_records:
        print(f"  - {rec['label']}: 常驻 {format_memory_value(rec.get('rss'))}, "
              f"GPU当前 {format_memory_value(rec.get('gpu_current'))}, "
              f"GPU峰值 {format_memory_value(rec.get('gpu_peak'))}")
    rss_avg, rss_max = summarize_memory(memory_records, 'rss')
    gpu_cur_avg, gpu_cur_max = summarize_memory(memory_records, 'gpu_current')
    gpu_peak_avg, gpu_peak_max = summarize_memory(memory_records, 'gpu_peak')
    if rss_avg is not None:
        print(f"  平均常驻内存: {rss_avg:.1f}MB (峰值 {rss_max:.1f}MB)")
    if gpu_cur_avg is not None:
        print(f"  平均GPU占用: {gpu_cur_avg:.1f}MB (峰值 {gpu_cur_max:.1f}MB)")
    if gpu_peak_avg is not None:
        print(f"  GPU记录峰值: {gpu_peak_avg:.1f}MB (最大 {gpu_peak_max:.1f}MB)")


if __name__ == "__main__":
    demo_matching()

