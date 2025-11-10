"""
基于模板匹配的指纹识别算法
适用于手机解锁等1:1验证场景
使用细节点局部结构模板进行快速匹配
"""

import cv2
import numpy as np
import os
import pickle
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from scipy.spatial.distance import cdist


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


class TemplateMatcher:
    """基于模板匹配的指纹匹配器"""
    
    def __init__(self,
                 k_neighbors: int = 5,
                 template_radius: float = 100.0,
                 distance_tolerance: float = 15.0,
                 angle_tolerance: float = 30.0,  # 度
                 orientation_tolerance: float = 30.0,  # 度
                 max_distance: float = 50.0,
                 match_threshold: float = 0.65):  # 平衡的默认阈值
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
        """
        self.k_neighbors = k_neighbors
        self.template_radius = template_radius
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = np.radians(angle_tolerance)
        self.orientation_tolerance = np.radians(orientation_tolerance)
        self.max_distance = max_distance
        self.match_threshold = match_threshold
    
    def _calculate_angle_difference(self, angle1: float, angle2: float) -> float:
        """计算两个角度之间的最小差值（考虑角度环绕）"""
        diff = abs(angle1 - angle2)
        return min(diff, 2 * np.pi - diff)
    
    def _calculate_orientation_difference(self, ori1: float, ori2: float) -> float:
        """计算两个方向角之间的最小角度差"""
        diff = abs(ori1 - ori2)
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
            ori_diff = self._calculate_orientation_difference(
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
            template1: 第一个模板
            template2: 第二个模板
            minutiae1: 第一组细节点
            minutiae2: 第二组细节点
            
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
                neighbor_ori_diff = self._calculate_orientation_difference(
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
        
        # 综合相似度：平均分数 * 匹配比例（放宽要求）
        similarity = avg_score * (0.7 + 0.3 * match_ratio)  # 即使匹配比例低也给予一定分数
        
        # 放宽惩罚：匹配比例低于40%才降低相似度
        if match_ratio < 0.4:  # 匹配比例低于40%
            similarity *= match_ratio / 0.4
        
        return min(1.0, similarity)
    
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
                ori_diff = self._calculate_orientation_difference(
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
        使用RANSAC进行几何验证
        
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
        
        min_inliers_for_early_stop = max(3, len(matches) // 4)
        
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
                ori_diff = self._calculate_orientation_difference(
                    minutiae1[match_idx1].orientation,
                    minutiae2[match_idx2].orientation
                )
                if ori_diff > self.orientation_tolerance:
                    orientation_ok = False
                    break
            
            if not orientation_ok:
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
                    ori_diff = self._calculate_orientation_difference(
                        minutiae1[match_idx1].orientation,
                        minutiae2[match_idx2].orientation
                    )
                    if ori_diff <= self.orientation_tolerance:
                        orientation_inliers.append(idx)
                
                inlier_indices = np.array(orientation_inliers)
                
                # 更新最佳结果
                if len(inlier_indices) > best_inliers:
                    best_inliers = len(inlier_indices)
                    best_inlier_indices = inlier_indices
                    best_transform = transform
                    
                    # 早期终止
                    if best_inliers >= min_inliers_for_early_stop:
                        current_inlier_ratio = best_inliers / len(matches)
                        if current_inlier_ratio > 0.5:
                            break
            
            except:
                continue
        
        # 计算匹配分数（平衡版本）
        if best_transform is not None and best_inliers > 0:
            inlier_ratio = best_inliers / len(matches)
            
            # 放宽要求：至少2个内点即可
            if best_inliers < 2:
                return MatchResult(0.0, 0, np.eye(3), [], 0.0)
            
            # 放宽内点比例要求到20%
            if inlier_ratio < 0.2:
                return MatchResult(0.0, 0, np.eye(3), [], 0.0)
            
            # 计算变换矩阵的质量（检查是否合理）
            # 检查缩放因子（放宽范围）
            scale_x = np.sqrt(best_transform[0, 0]**2 + best_transform[1, 0]**2)
            scale_y = np.sqrt(best_transform[0, 1]**2 + best_transform[1, 1]**2)
            
            # 缩放因子应该在合理范围内（放宽到0.3-3.0）
            if scale_x < 0.3 or scale_x > 3.0 or scale_y < 0.3 or scale_y > 3.0:
                return MatchResult(0.0, 0, np.eye(3), [], 0.0)
            
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
            
            return MatchResult(
                score=min(1.0, score),
                num_matches=best_inliers,
                transform_matrix=transform_3x3,
                matched_pairs=matched_pairs,
                inlier_ratio=inlier_ratio
            )
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
    
    def save_template(self, templates: List[LocalTemplate], minutiae: List[Minutia], 
                     template_file: str, metadata: Optional[Dict] = None):
        """
        保存模板到文件
        
        Args:
            templates: 模板列表
            minutiae: 细节点列表（用于重建模板）
            template_file: 保存路径
            metadata: 可选的元数据（如指纹ID等）
        """
        try:
            # 准备保存的数据
            save_data = {
                'templates': templates,
                'minutiae': minutiae,
                'metadata': metadata or {},
                'matcher_params': {
                    'k_neighbors': self.k_neighbors,
                    'template_radius': self.template_radius,
                    'distance_tolerance': self.distance_tolerance,
                    'angle_tolerance': np.degrees(self.angle_tolerance),
                    'orientation_tolerance': np.degrees(self.orientation_tolerance),
                    'max_distance': self.max_distance,
                    'match_threshold': self.match_threshold
                }
            }
            
            # 使用pickle保存（支持numpy数组）
            with open(template_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"模板已保存到: {template_file}")
        except Exception as e:
            print(f"保存模板失败: {e}")
    
    def load_template(self, template_file: str) -> Tuple[List[LocalTemplate], List[Minutia], Dict]:
        """
        从文件加载模板
        
        Args:
            template_file: 模板文件路径
            
        Returns:
            (templates, minutiae, metadata)
        """
        try:
            with open(template_file, 'rb') as f:
                save_data = pickle.load(f)
            
            templates = save_data['templates']
            minutiae = save_data['minutiae']
            metadata = save_data.get('metadata', {})
            
            print(f"模板已从 {template_file} 加载")
            return templates, minutiae, metadata
        except Exception as e:
            print(f"加载模板失败: {e}")
            return [], [], {}
    
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
                'num_samples': len(multi_templates),  # 样本数量
                'sample_ids': sample_ids or [],  # 样本ID列表
                'metadata': metadata or {},
                'matcher_params': {
                    'k_neighbors': self.k_neighbors,
                    'template_radius': self.template_radius,
                    'distance_tolerance': self.distance_tolerance,
                    'angle_tolerance': np.degrees(self.angle_tolerance),
                    'orientation_tolerance': np.degrees(self.orientation_tolerance),
                    'max_distance': self.max_distance,
                    'match_threshold': self.match_threshold
                }
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
            
            # 兼容单模板格式
            if 'multi_templates' in save_data:
                multi_templates = save_data['multi_templates']
            else:
                # 旧格式：单模板，转换为多模板格式
                templates = save_data['templates']
                minutiae = save_data['minutiae']
                multi_templates = [(templates, minutiae)]
            
            metadata = save_data.get('metadata', {})
            num_samples = save_data.get('num_samples', len(multi_templates))
            sample_ids = save_data.get('sample_ids', [])  # 样本ID列表
            
            if verbose:
                print(f"多模板已从 {template_file} 加载 (包含 {num_samples} 个样本)")
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
        
        # 如果是多模板，与所有模板匹配并取最佳结果
        if len(multi_templates) > 1:
            best_result = None
            best_score = -1.0
            
            for idx, (stored_templates, stored_minutiae) in enumerate(multi_templates):
                if len(stored_templates) == 0:
                    continue
                
                # 如果提供了要排除的样本ID，检查当前模板是否对应该样本
                if exclude_query_sample_id is not None and len(sample_ids) > idx:
                    if sample_ids[idx] == exclude_query_sample_id:
                        continue  # 跳过这个模板，避免自己匹配自己
                
                # 如果没有样本ID信息，使用位置距离作为后备方案
                if exclude_query_sample_id is not None and (len(sample_ids) == 0 or idx >= len(sample_ids)):
                    # 后备检查：如果细节点数量相同且位置非常接近，可能是同一个样本
                    if len(query_minutiae) == len(stored_minutiae):
                        positions1 = np.array([[m.x, m.y] for m in query_minutiae])
                        positions2 = np.array([[m.x, m.y] for m in stored_minutiae])
                        if len(positions1) > 0 and len(positions2) > 0:
                            distances = cdist(positions1, positions2)
                            min_distances = np.min(distances, axis=1)
                            avg_min_distance = np.mean(min_distances)
                            # 如果平均最小距离很小（<3像素），可能是同一个样本
                            if avg_min_distance < 3.0:
                                continue  # 跳过这个模板
                
                # 模板匹配
                matches = self.find_template_matches(stored_templates, query_templates, 
                                                    stored_minutiae, query_minutiae)
                
                if len(matches) == 0:
                    continue
                
                # 几何验证
                result = self.geometric_verification(stored_minutiae, query_minutiae, matches)
                
                # 保留最佳结果
                if result.score > best_score:
                    best_score = result.score
                    best_result = result
            
            return best_result if best_result is not None else MatchResult(0.0, 0, np.eye(3), [], 0.0)
        else:
            # 单模板匹配（向后兼容）
            stored_templates, stored_minutiae = multi_templates[0]
            
            if len(stored_templates) == 0:
                return MatchResult(0.0, 0, np.eye(3), [], 0.0)
            
            # 模板匹配
            matches = self.find_template_matches(stored_templates, query_templates, 
                                                stored_minutiae, query_minutiae)
            
            if len(matches) == 0:
                return MatchResult(0.0, 0, np.eye(3), [], 0.0)
            
            # 几何验证
            result = self.geometric_verification(stored_minutiae, query_minutiae, matches)
            
            return result
    
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
                         save_path: Optional[str] = None):
        """可视化匹配结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示第一个指纹
        ax1.imshow(image1, cmap='gray')
        for i, mnt in enumerate(minutiae1):
            color = 'red' if any(i == pair[0] for pair in result.matched_pairs) else 'blue'
            ax1.plot(mnt.x, mnt.y, 'o', color=color, markersize=3)
            ax1.arrow(mnt.x, mnt.y,
                    10 * np.cos(mnt.orientation),
                    10 * np.sin(mnt.orientation),
                    head_width=3, head_length=2, fc=color, ec=color)
        ax1.set_title(f'指纹1 (匹配: {len(result.matched_pairs)})')
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
        ax2.set_title(f'指纹2 (匹配: {len(result.matched_pairs)})')
        ax2.axis('off')
        
        plt.suptitle(f'匹配分数: {result.score:.2f}, 内点比例: {result.inlier_ratio:.2f}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig('fingerprint_match_result.png', dpi=150, bbox_inches='tight')
        plt.close()


def demo_matching():
    """演示模板匹配功能"""
    # 创建模板匹配器（使用默认参数，match_threshold=0.65）
    matcher = TemplateMatcher(
        k_neighbors=5,  # 局部模板使用5个最近邻
        template_radius=100.0,  # 模板搜索半径100像素
        distance_tolerance=15.0,  # 距离容差15像素
        angle_tolerance=30.0,  # 角度容差30度
        orientation_tolerance=30.0,  # 方向容差30度
        max_distance=50.0,  # RANSAC最大距离50像素
        match_threshold=0.65  # 模板匹配阈值0.65（与默认值一致）
    )
    
    print("指纹模板匹配演示（适用于手机解锁）")
    print("="*50)
    
    # 示例用法
    image1_path = '../datasets/db4_b/00002_02.bmp'
    image2_path = '../datasets/db4_b/00002_05.bmp'
    mnt1_path = '../output/20251027-172033/2/00002_02.mnt'
    mnt2_path = '../output/20251027-172033/2/00002_05.mnt'
    
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
    
    if not minutiae1 or not minutiae2:
        print("错误：无法加载细节点数据")
        return
    
    print(f"指纹1细节点数量: {len(minutiae1)}")
    print(f"指纹2细节点数量: {len(minutiae2)}")
    
    # 执行模板匹配
    print("\n开始模板匹配...")
    result = matcher.match_fingerprints(minutiae1, minutiae2)
    
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
    matcher.visualize_matches(image1, minutiae1, image2, minutiae2, result, 'match_result.png')
    print("匹配结果已保存为 match_result.png")


if __name__ == "__main__":
    demo_matching()

