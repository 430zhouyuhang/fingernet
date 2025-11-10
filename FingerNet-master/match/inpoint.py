"""
基于描述符的快速指纹匹配算法
使用FingerNet输出的细节点特征进行指纹匹配
"""

import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Tcl/Tk依赖
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from scipy.spatial.distance import cdist


@dataclass
class Minutia:
    """细节点数据结构"""
    x: float
    y: float
    orientation: float  # 弧度
    confidence: float
    minutia_type: int = 0  # 0: 端点, 1: 分叉点  特征提取中无这个部分


@dataclass
class MatchResult:
    """匹配结果"""
    score: float
    num_matches: int
    transform_matrix: np.ndarray
    matched_pairs: List[Tuple[int, int]]
    inlier_ratio: float


class FingerprintMatcher:
    """基于描述符的指纹匹配器"""
    
    def __init__(self, 
                 descriptor_radius: int = 16,
                 match_threshold: float = 0.7,
                 max_distance: float = 50.0,
                 orientation_tolerance: float = 30.0):  # 度
        """
        初始化匹配器
        
        Args:
            descriptor_radius: 描述符提取半径
            match_threshold: 匹配阈值
            max_distance: 最大匹配距离
            orientation_tolerance: 方向容差(度)
        """
        self.descriptor_radius = descriptor_radius
        self.match_threshold = match_threshold
        self.max_distance = max_distance
        self.orientation_tolerance = np.radians(orientation_tolerance)
        
        # 初始化SIFT描述符提取器
        self.sift = cv2.SIFT_create()
        
    def _calculate_orientation_difference(self, ori1: float, ori2: float) -> float:
        """
        计算两个方向角之间的最小角度差（考虑角度环绕）
        
        Args:
            ori1: 第一个方向角（弧度）
            ori2: 第二个方向角（弧度）
            
        Returns:
            最小角度差（弧度）
        """
        diff = abs(ori1 - ori2)
        # 处理角度环绕：取较小的角度差
        return min(diff, 2 * np.pi - diff)
    
    
    def is_match_successful(self, score: float, threshold: float = 0.5) -> bool:
        """
        判断匹配是否成功
        
        Args:
            score: 匹配分数 (0-1)
            threshold: 匹配阈值 (默认0.5)
            
        Returns:
            是否匹配成功
        """
        return score >= threshold
    
    def get_score_interpretation(self, score: float) -> str:
        """
        获取分数解释
        
        Args:
            score: 匹配分数 (0-1)
            
        Returns:
            分数解释字符串
        """
        if score >= 0.5:
            return "匹配成功 - 是同一指纹"
        else:
            return "匹配失败 - 非同一指纹"
    
    def load_minutiae_from_mnt(self, mnt_file: str) -> List[Minutia]:
        """
        从.mnt文件加载细节点数据
        
        Args:
            mnt_file: .mnt文件路径
            
        Returns:
            细节点列表
        """
        minutiae = []
        try:
            with open(mnt_file, 'r') as f:
                lines = f.readlines()
                
            # 跳过前两行（文件头）
            for line in lines[2:]:
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
    
    
    def extract_local_descriptor(self, 
                                image: np.ndarray, 
                                minutia: Minutia) -> np.ndarray:
        """
        提取细节点局部描述符
        
        Args:
            image: 指纹图像
            minutia: 细节点
            
        Returns:
            描述符向量
        """
        x, y = int(minutia.x), int(minutia.y)
        radius = self.descriptor_radius
        
        # 使用固定半径提取描述符
        adjusted_radius = radius
        
        # 确保坐标在图像范围内
        h, w = image.shape
        x1 = max(0, x - adjusted_radius)
        y1 = max(0, y - adjusted_radius)
        x2 = min(w, x + adjusted_radius + 1)
        y2 = min(h, y + adjusted_radius + 1)
        
        # 提取局部区域
        patch = image[y1:y2, x1:x2]
        
        if patch.size == 0:
            return np.zeros(128)  # SIFT描述符维度
        
        # 创建关键点
        kp = cv2.KeyPoint(
            x - x1, y - y1,  # 相对坐标
            size=adjusted_radius * 2,
            angle=np.degrees(minutia.orientation),
            response=1.0  # 固定响应值
        )
        
        # 计算SIFT描述符
        try:
            _, descriptor = self.sift.compute(patch, [kp])
            if descriptor is not None and len(descriptor) > 0:
                # 保持描述符的原始归一化特性
                return descriptor[0]
        except:
            pass
            
        return np.zeros(128)
    
    def compute_descriptors(self, 
                           image: np.ndarray, 
                           minutiae: List[Minutia]) -> np.ndarray:
        """
        计算所有细节点的描述符
        
        Args:
            image: 指纹图像
            minutiae: 细节点列表
            
        Returns:
            描述符矩阵 (N, 128)
        """
        descriptors = []
        for minutia in minutiae:
            desc = self.extract_local_descriptor(image, minutia)
            descriptors.append(desc)
        
        return np.array(descriptors)
    
    def match_descriptors(self, 
                         desc1: np.ndarray, 
                         desc2: np.ndarray,
                         minutiae1: List[Minutia] = None,
                         minutiae2: List[Minutia] = None) -> List[Tuple[int, int, float]]:
        """
        匹配两组描述符
        Args:
            desc1: 第一组描述符
            desc2: 第二组描述符
            minutiae1: 第一组细节点（用于置信度加权和方向验证）
            minutiae2: 第二组细节点（用于置信度加权和方向验证）
            
        Returns:
            匹配对列表 (idx1, idx2, similarity)
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # 计算描述符间的余弦相似度
        # 归一化描述符
        desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
        desc2_norm = desc2 / (np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-8)
        
        # 计算相似度矩阵
        similarity_matrix = np.dot(desc1_norm, desc2_norm.T)
        
        # 确保相似度矩阵在合理范围内
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        
        # 使用贪心算法进行最优匹配（添加方向验证）
        matches = []
        used_indices_2 = set()  # 记录已使用的desc2索引
        
        if len(desc1) <= len(desc2):
            # 为每个desc1找到最佳匹配
            for i in range(len(desc1)):
                best_j = -1
                best_sim = 0.0
                
                for j in range(len(desc2)):
                    if j in used_indices_2:
                        continue
                        
                    sim = similarity_matrix[i, j]
                    if sim > best_sim:
                        # 检查方向一致性
                        if minutiae1 is not None and minutiae2 is not None:
                            ori_diff = self._calculate_orientation_difference(
                                minutiae1[i].orientation, 
                                minutiae2[j].orientation
                            )
                            if ori_diff <= self.orientation_tolerance:
                                best_j = j
                                best_sim = sim
                        else:
                            # 如果没有细节点信息，跳过方向验证
                            best_j = j
                            best_sim = sim
                
                if best_j != -1 and best_sim > self.match_threshold:
                    matches.append((i, best_j, best_sim))
                    used_indices_2.add(best_j)
        else:
            # 为每个desc2找到最佳匹配
            used_indices_1 = set()  # 记录已使用的desc1索引
            
            for j in range(len(desc2)):
                best_i = -1
                best_sim = 0.0
                
                for i in range(len(desc1)):
                    if i in used_indices_1:
                        continue
                        
                    sim = similarity_matrix[i, j]
                    if sim > best_sim:
                        # 检查方向一致性
                        if minutiae1 is not None and minutiae2 is not None:
                            ori_diff = self._calculate_orientation_difference(
                                minutiae1[i].orientation, 
                                minutiae2[j].orientation
                            )
                            if ori_diff <= self.orientation_tolerance:
                                best_i = i
                                best_sim = sim
                        else:
                            # 如果没有细节点信息，跳过方向验证
                            best_i = i
                            best_sim = sim
                
                if best_i != -1 and best_sim > self.match_threshold:
                    matches.append((best_i, j, best_sim))
                    used_indices_1.add(best_i)
        
        return matches
    
    def geometric_verification(self, 
                              minutiae1: List[Minutia],
                              minutiae2: List[Minutia],
                              matches: List[Tuple[int, int, float]]) -> MatchResult:
        """
        几何验证和变换估计
        
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
        
        # 使用RANSAC估计仿射变换
        best_transform = None
        best_inliers = 0
        best_inlier_indices = []
        
        # 内点阈值
        inlier_threshold = self.max_distance
        
        # 自适应迭代次数计算
        # 根据内点比例动态调整，但设置合理的上下限
        estimated_inlier_ratio = 0.3  # 保守估计30%内点
        max_iterations = min(1000, max(100, int(np.log(0.01) / np.log(1 - estimated_inlier_ratio**3))))
        
        # 早期终止条件
        min_inliers_for_early_stop = max(3, len(matches) // 4)
        
        # 调试信息
        print(f"RANSAC参数: 最大迭代={max_iterations}, 内点阈值={inlier_threshold:.1f}px, 方向容差={np.degrees(self.orientation_tolerance):.1f}°")
        
        # 统计信息
        successful_iterations = 0
        orientation_rejected = 0
        
        for iteration in range(max_iterations):
            if len(matches) < 3:
                break
                
            # 随机选择3个点
            sample_indices = np.random.choice(len(matches), 3, replace=False)
            sample_pts1 = pts1[sample_indices]
            sample_pts2 = pts2[sample_indices]
            
            # 检查采样点的方向一致性（使用统一标准）
            orientation_ok = True
            for idx in sample_indices:
                match_idx1, match_idx2 = matches[idx][0], matches[idx][1]
                ori_diff = self._calculate_orientation_difference(
                    minutiae1[match_idx1].orientation,
                    minutiae2[match_idx2].orientation
                )
                # 使用统一的方向容差标准
                if ori_diff > self.orientation_tolerance:
                    orientation_ok = False
                    break
            
            if not orientation_ok:
                orientation_rejected += 1
                continue
            
            try:
                # 估计仿射变换
                transform = cv2.getAffineTransform(
                    sample_pts1.astype(np.float32),
                    sample_pts2.astype(np.float32)
                )
                
                # 计算内点（几何验证）
                ones = np.ones((len(pts1), 1))
                pts1_homo = np.hstack([pts1, ones])
                transform_3x3 = np.vstack([transform, [0, 0, 1]])
                transformed_pts1 = np.dot(pts1_homo, transform_3x3.T)[:, :2]
                
                distances = np.linalg.norm(transformed_pts1 - pts2, axis=1)
                geometric_inliers = np.where(distances < inlier_threshold)[0]
                
                # 进一步检查方向一致性（在几何内点中筛选）
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
                    successful_iterations += 1
                    
                    # 早期终止：如果找到足够多的内点
                    if best_inliers >= min_inliers_for_early_stop:
                        # 重新估计内点比例，动态调整迭代次数
                        current_inlier_ratio = best_inliers / len(matches)
                        if current_inlier_ratio > 0.5:  # 如果内点比例很高，可以提前终止
                            print(f"早期终止: 迭代{iteration+1}, 内点{best_inliers}, 比例{current_inlier_ratio:.2f}")
                            break
                    
            except:
                continue
        
        # 输出RANSAC统计信息
        print(f"RANSAC完成: 总迭代{iteration+1}, 成功{successful_iterations}, 方向拒绝{orientation_rejected}")
        
        # 计算匹配分数
        if best_transform is not None and best_inliers > 0:
            inlier_ratio = best_inliers / len(matches)
            
            # 使用简化的匹配分数计算（不使用置信度）
            # score = self._calculate_match_score(best_inliers, inlier_ratio, 1.0, len(matches))
            score = inlier_ratio  # 直接使用内点比例
            
            # 转换为3x3齐次变换矩阵
            transform_3x3 = np.eye(3)
            transform_3x3[:2, :] = best_transform
            
            matched_pairs = [(matches[i][0], matches[i][1]) for i in best_inlier_indices]
            
            return MatchResult(
                score=score,
                num_matches=best_inliers,
                transform_matrix=transform_3x3,
                matched_pairs=matched_pairs,
                inlier_ratio=inlier_ratio
            )
        else:
            return MatchResult(0.0, 0, np.eye(3), [], 0.0)
    
    def match_fingerprints(self, 
                          image1: np.ndarray,
                          minutiae1: List[Minutia],
                          image2: np.ndarray,
                          minutiae2: List[Minutia]) -> MatchResult:
        """
        匹配两个指纹
        
        Args:
            image1: 第一个指纹图像
            minutiae1: 第一个指纹的细节点
            image2: 第二个指纹图像
            minutiae2: 第二个指纹的细节点
            
        Returns:
            匹配结果
        """
        # 计算描述符
        desc1 = self.compute_descriptors(image1, minutiae1)
        desc2 = self.compute_descriptors(image2, minutiae2)
        
        # 匹配描述符
        matches = self.match_descriptors(desc1, desc2, minutiae1, minutiae2)
        
        if len(matches) == 0:
            return MatchResult(0.0, 0, np.eye(3), [], 0.0)
        
        # 几何验证
        result = self.geometric_verification(minutiae1, minutiae2, matches)
        
        return result
    
    def visualize_matches(self,
                         image1: np.ndarray,
                         minutiae1: List[Minutia],
                         image2: np.ndarray,
                         minutiae2: List[Minutia],
                         result: MatchResult,
                         save_path: Optional[str] = None):
        """
        可视化匹配结果
        
        Args:
            image1: 第一个指纹图像
            minutiae1: 第一个指纹的细节点
            image2: 第二个指纹图像
            minutiae2: 第二个指纹的细节点
            result: 匹配结果
            save_path: 保存路径
        """
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
            # 自动保存到当前目录
            plt.savefig('fingerprint_match_result.png', dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存


def demo_matching():
    """演示匹配功能"""
    # 创建匹配器
    matcher = FingerprintMatcher(
        descriptor_radius=16,  #SIFT描述符提取的基础半径（像素）
        match_threshold=0.6,   #匹配阈值，只有相似度 > 0.6 的匹配对才会被保留
        max_distance=50.0,     #在RANSAC几何验证中，判断点是否为内点，超过这个距离的匹配对会被忽略
        orientation_tolerance=30.0 #在几何验证中，允许的细节点方向差异（度），超过这个角度的匹配对会被忽略
    )
    
    print("指纹匹配演示")
    print("="*50)
    
    # 示例用法
    # 加载图像和细节点
    image1_path = '../datasets/db4_b/00002_06.bmp'
    image2_path = '../datasets/db4_b/00000_02.bmp'
    mnt1_path = '../output/20251021-141618/2/00002_06.mnt'
    mnt2_path = '../output/20251021-141618/2/00000_02.mnt'
    
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
    
    # 显示细节点统计
    print(f"指纹1细节点数量: {len(minutiae1)}")
    print(f"指纹2细节点数量: {len(minutiae2)}")
    
    # 直接使用所有细节点（不进行置信度过滤）
    minutiae1_filtered = minutiae1
    minutiae2_filtered = minutiae2
    
    # 执行匹配（使用过滤后的细节点）
    print("\n开始匹配...")
    result = matcher.match_fingerprints(image1, minutiae1_filtered, image2, minutiae2_filtered)
    
    # 输出结果
    print(f"\n匹配结果:")
    print(f"匹配分数: {result.score:.3f} (归一化 0-1)")
    print(f"匹配数量: {result.num_matches}")
    print(f"内点比例: {result.inlier_ratio:.2f}")
    print(f"方向容差: {np.degrees(matcher.orientation_tolerance):.1f}度")
    
    # 匹配成功判断
    match_threshold = 0.5  # 推荐阈值
    is_success = matcher.is_match_successful(result.score, match_threshold)
    print(f"\n匹配成功判断:")
    print(f"  阈值: {match_threshold}")
    print(f"  结果: {'✓ 成功' if is_success else '✗ 失败'}")
    
    # 分数解释
    interpretation = matcher.get_score_interpretation(result.score)
    print(f"\n分数解释: {interpretation}")
    
    # 可视化（自动保存图片）
    print("\n正在生成匹配结果可视化图片...")
    matcher.visualize_matches(image1, minutiae1_filtered, image2, minutiae2_filtered, result, 'match_result.png')
    print("匹配结果已保存为 match_result.png")

if __name__ == "__main__":
    demo_matching()
