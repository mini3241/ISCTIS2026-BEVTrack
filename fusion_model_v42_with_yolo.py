import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import cv2
from sklearn.cluster import DBSCAN  # 🔴 新增：DBSCAN聚类
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║  🔴 V42增强版 Fusion Model - DBSCAN + YOLO集成 (对齐Baseline)                      ║
# ║  fusion_model_v42_with_yolo.py                                                    ║
# ╠═══════════════════════════════════════════════════════════════════════════════════╣
# ║  V42新增 (YOLO集成 - 对齐Baseline RadarRGBFusionNet2_20231128):                    ║
# ║  1. YOLODetector: YOLOv5m检测器 (只检测car, class=2)                              ║
# ║  2. FusionState: 检测源状态枚举 (0=无, 1=相机, 2=雷达, 3=融合)                     ║
# ║  3. Detection: 检测结果数据类                                                      ║
# ║  4. YOLOIntegrationManager: YOLO集成管理器                                        ║
# ║  5. fuse_detection_positions: 多源位置融合函数                                     ║
# ║                                                                                   ║
# ║  原有功能:                                                                         ║
# ║  - DBSCAN雷达点云去噪 (eps=1.0, min_samples=8)                                    ║
# ║  - ReIDExtractor外观特征提取                                                       ║
# ║  - KalmanMOTATracker卡尔曼跟踪器                                                   ║
# ║                                                                                   ║
# ║  BEV范围: [-10,10]×[0,50]m (与Baseline一致)                                       ║
# ║  SGDNet输入: 960×256                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════════════════════════════

# DBSCAN参数（与Baseline一致: eps=0.9, min_samples=7）
DBSCAN_EPS = 0.9  # 邻域半径 (Baseline=0.9)
DBSCAN_MIN_SAMPLES = 7  # 最小样本数 (Baseline=7)
BEV_X_MIN, BEV_X_MAX = -10.0, 10.0
BEV_Y_MIN, BEV_Y_MAX = 5.0, 40.0  # 对齐Baseline OCULii
USE_DBSCAN = True  # 🔴 全局开关：是否使用DBSCAN去噪
USE_COORD_SWAP = False  # 🔴 V42数据加载时已做坐标交换，DBSCAN处不需要再做

# YOLO配置（对齐Baseline）
ENABLE_YOLO = True  # 🔴 全局开关：是否使用YOLO
YOLO_MODEL_NAME = 'yolov5m'  # Baseline用yolov5m
YOLO_CONF_THRESHOLD = 0.3
YOLO_IOU_THRESHOLD = 0.5
YOLO_CLASSES = [2]  # 🔴 只检测car (COCO class 2)
YOLO_ENABLE_REID = True
YOLO_HALF = True  # FP16加速
YOLO_AUGMENT = True  # 增强推理

# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 V42+ 特征级融合模式配置
# ═══════════════════════════════════════════════════════════════════════════════════
# 在特征级融合架构中：
#   - 位置检测：由融合heatmap + DBSCAN完成（不需要YOLO位置）
#   - ReID特征：由YOLO bbox crop + OSNet提取（用于跟踪）
#   - 原始雷达DBSCAN：冗余（雷达已在特征级融合）
#
# YOLO_REID_ONLY=True时：
#   - YOLO只用于提取ReID特征，不参与位置融合
#   - 未匹配的YOLO检测不作为独立检测结果
#   - 位置完全来自融合heatmap的DBSCAN结果
# ═══════════════════════════════════════════════════════════════════════════════════
YOLO_REID_ONLY = False  # 🔴 特征级融合模式：YOLO只用于ReID特征提取，不融合位置
USE_RAW_RADAR_DBSCAN = False  # 🔴 特征级融合模式：禁用原始雷达点云DBSCAN（冗余）

# 相机内参（默认值，可在初始化时覆盖）
DEFAULT_CAMERA_MATRIX = {
    'fx': 1000.0, 'fy': 1000.0,
    'cx': 480.0, 'cy': 128.0,  # 960×256图像中心
}

print(f"✅ V42 Fusion Model 已加载 (对齐Baseline)")
print(f"   DBSCAN: {'启用' if USE_DBSCAN else '禁用'} (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})")
print(f"   坐标重排(Y↔Z): {'启用' if USE_COORD_SWAP else '禁用 (数据加载时已做)'}")
print(f"   YOLO: {'启用' if ENABLE_YOLO else '禁用'} (model={YOLO_MODEL_NAME}, classes={YOLO_CLASSES})")
print(f"   🔴 特征级融合模式:")
print(f"      YOLO_REID_ONLY={YOLO_REID_ONLY} (YOLO只用于ReID，不融合位置)")
print(f"      USE_RAW_RADAR_DBSCAN={USE_RAW_RADAR_DBSCAN} (原始雷达DBSCAN)")


# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 V42新增: FusionState枚举 (对齐Baseline Detection类)
# ═══════════════════════════════════════════════════════════════════════════════════

class FusionState(IntEnum):
    """
    检测源状态标记 (对齐Baseline Detection.fusion_state)

    Baseline定义 (Detection/detection.py):
        0: no target
        1: only_camera
        2: only_radar
        3: fusion_sensors
    """
    NO_TARGET = 0
    CAMERA_ONLY = 1  # 只有YOLO检测到
    RADAR_ONLY = 2  # 只有热力图/雷达检测到
    FUSION = 3  # 雷达+相机都检测到（融合）


@dataclass
class Detection:
    """
    检测结果数据类 (对齐Baseline Detection类)

    包含:
    - center: BEV世界坐标 (x, y)
    - fusion_state: 检测源状态
    - confidence: 检测置信度
    - feature: 512维ReID特征
    - tlwh: 2D bbox [top, left, width, height] (YOLO输出)
    - proj_xy: 投影坐标
    - r_center: 雷达中心坐标
    - c_center: 相机中心坐标
    """
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    fusion_state: int = FusionState.RADAR_ONLY
    confidence: float = 0.0
    feature: np.ndarray = field(default_factory=lambda: np.zeros(512))
    tlwh: np.ndarray = field(default_factory=lambda: np.zeros(4))
    proj_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    classes: int = 2  # 默认car
    r_center: Optional[np.ndarray] = None  # 雷达检测中心
    c_center: Optional[np.ndarray] = None  # 相机检测中心
    depth: float = 0.0

    def to_tuple(self) -> Tuple[float, float]:
        """转换为(x, y)元组格式"""
        return (float(self.center[0]), float(self.center[1]))

    def to_tlbr(self) -> np.ndarray:
        """转换为[x1, y1, x2, y2]格式"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 V42新增: YOLODetector类 (对齐Baseline YOLO_reid_model)
# ═══════════════════════════════════════════════════════════════════════════════════

class YOLODetector:
    """
    YOLOv5检测器 (对齐Baseline Detection/detection.py YOLO_reid_model)

    Baseline配置:
    - 模型: yolov5m.pt
    - 类别: classes=2 (只检测car)
    - 置信度: conf_thres=0.3
    - IOU: iou_thres=0.5

    功能:
    1. 2D目标检测
    2. 深度查询（从SGDNet深度图）
    3. 2D→BEV投影
    4. ReID特征提取
    """

    def __init__(
            self,
            model_name: str = 'yolov5m',
            pretrained: bool = True,
            device: str = 'cuda',
            conf_threshold: float = 0.3,
            iou_threshold: float = 0.5,
            classes: List[int] = None,
            camera_matrix: Dict = None,
            enable_reid: bool = True,
            half: bool = True,
            augment: bool = True,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or [2]  # 默认只检测car
        self.camera_matrix = camera_matrix or DEFAULT_CAMERA_MATRIX
        self.enable_reid = enable_reid
        self.half = half
        self.augment = augment

        self.model = None
        self.reid_extractor = None
        self.is_loaded = False

        self._load_model(model_name, pretrained)

    def _load_model(self, model_name: str, pretrained: bool):
        """加载YOLOv5模型"""
        try:
            # 🔧 使用项目内的yolov5仓库（不依赖ultralytics）
            project_yolov5_repo = '/mnt/ChillDisk/personal_data/mij/pythonProject1/RadarRGBFusionNet2_20231128/Detection/yolov5'
            local_yolo_weights = '/mnt/ChillDisk/personal_data/mij/pythonProject1/yolov5m.pt'

            # 检查本地权重文件
            if not os.path.exists(local_yolo_weights):
                # 尝试备选路径
                alt_weights = '/mnt/ChillDisk/personal_data/mij/pythonProject1/RadarRGBFusionNet2_20231128/Detection/yolov5/weights/yolov5m.pt'
                if os.path.exists(alt_weights):
                    local_yolo_weights = alt_weights
                else:
                    local_yolo_weights = None

            # 方法1: 使用项目内yolov5仓库 + 本地权重
            if os.path.exists(project_yolov5_repo):
                print(f"🔄 从项目内yolov5仓库加载: {project_yolov5_repo}")
                try:
                    if local_yolo_weights and os.path.exists(local_yolo_weights):
                        # 加载本地权重文件
                        print(f"   使用本地权重: {local_yolo_weights}")
                        self.model = torch.hub.load(
                            project_yolov5_repo, 'custom',
                            path=local_yolo_weights,
                            source='local'
                        )
                    else:
                        # 从网络下载预训练权重
                        print(f"   下载预训练权重: {model_name}")
                        self.model = torch.hub.load(
                            project_yolov5_repo, model_name,
                            source='local',
                            pretrained=pretrained
                        )
                except Exception as e_local:
                    print(f"⚠️ 项目yolov5加载失败: {e_local}")
                    # 方法2: 尝试从网络加载
                    print(f"🔄 尝试从网络加载...")
                    self.model = torch.hub.load(
                        'ultralytics/yolov5', model_name,
                        pretrained=pretrained, trust_repo=True,
                        force_reload=False
                    )
            else:
                # 方法2: 项目内仓库不存在，从网络加载
                print(f"🔄 项目yolov5仓库不存在，从网络加载...")
                self.model = torch.hub.load(
                    'ultralytics/yolov5', model_name,
                    pretrained=pretrained, trust_repo=True,
                    force_reload=False
                )
            self.model.to(self.device)
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            self.model.classes = self.classes

            # FP16加速
            if self.half and self.device != 'cpu':
                self.model.half()

            self.is_loaded = True
            print(f"✅ YOLO检测器加载成功: {model_name}")
            print(f"   类别过滤: {self.classes} (2=car)")
            print(f"   置信度: {self.conf_threshold}, IOU: {self.iou_threshold}")
        except Exception as e:
            print(f"⚠️ YOLO加载失败: {e}")
            print("   将使用模拟检测（返回空列表）")
            self.is_loaded = False

        if self.enable_reid:
            self._load_reid()

    def _load_reid(self):
        """加载ReID特征提取器 (对齐Baseline OSNet)"""
        try:
            # 尝试加载OSNet
            try:
                sys.path.insert(0, 'Detection/deep/reid')
                from torchreid import models as reid_models
                self.reid_extractor = reid_models.build_model(
                    name='osnet_ain_x1_0', num_classes=1000
                )
                self.reid_extractor.to(self.device)
                self.reid_extractor.eval()
                print("✅ ReID: OSNet加载成功 (对齐Baseline)")
            except:
                # 使用ResNet18作为备选
                from torchvision import models
                resnet = models.resnet18(pretrained=True)
                self.reid_extractor = nn.Sequential(*list(resnet.children())[:-1])
                self.reid_extractor.to(self.device)
                self.reid_extractor.eval()
                print("✅ ReID: ResNet18加载成功 (备选)")
        except Exception as e:
            print(f"⚠️ ReID加载失败: {e}")
            self.reid_extractor = None

    def detect(
            self,
            image: np.ndarray,
            depth_map: np.ndarray = None
    ) -> List[Detection]:
        """
        运行YOLO检测并转换到BEV坐标

        Args:
            image: RGB图像 (H,W,3) 或 (3,H,W)
            depth_map: SGDNet深度图 (H,W)，用于2D→3D投影

        Returns:
            检测列表（包含BEV坐标和ReID特征）
        """
        if not self.is_loaded or self.model is None:
            return []

        # 格式转换
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        # 确保图像是uint8格式
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # YOLO推理
        with torch.no_grad():
            results = self.model(image, augment=self.augment)

        # 解析结果
        detections = []
        if hasattr(results, 'xyxy'):
            boxes = results.xyxy[0].cpu().numpy()
        elif hasattr(results, 'pred'):
            boxes = results.pred[0].cpu().numpy()
        else:
            return []

        for box in boxes:
            if len(box) < 6:
                continue
            x1, y1, x2, y2, conf, cls = box[:6]

            # 计算2D中心和bbox
            cx_2d = (x1 + x2) / 2
            cy_2d = (y1 + y2) / 2
            w_2d = x2 - x1
            h_2d = y2 - y1

            # 深度查询（从SGDNet深度图）
            if depth_map is not None:
                h, w = depth_map.shape[:2]
                px = int(np.clip(cx_2d, 0, w - 1))
                py = int(np.clip(cy_2d, 0, h - 1))
                depth = float(depth_map[py, px])
                depth = np.clip(depth, 1.0, 50.0)
            else:
                # 无深度图时根据bbox大小估计
                depth = np.clip(800.0 / (h_2d + 1e-6), 1.0, 50.0)

            # 2D→BEV投影
            world_x, world_y = self._project_to_bev(cx_2d, cy_2d, depth)

            # 提取ReID特征
            feature = np.zeros(512, dtype=np.float32)
            if self.enable_reid and self.reid_extractor is not None:
                feature = self._extract_reid(image, [x1, y1, x2, y2])

            det = Detection(
                center=np.array([world_x, world_y]),
                fusion_state=FusionState.CAMERA_ONLY,
                confidence=float(conf),
                feature=feature,
                tlwh=np.array([y1, x1, w_2d, h_2d]),  # top, left, width, height
                proj_xy=np.array([cx_2d, cy_2d]),
                classes=int(cls),
                c_center=np.array([world_x, world_y]),
                depth=depth,
            )
            detections.append(det)

        return detections

    def _project_to_bev(
            self,
            cx_2d: float,
            cy_2d: float,
            depth: float
    ) -> Tuple[float, float]:
        """
        2D图像坐标→BEV世界坐标

        相机坐标系: X右, Y下, Z前
        BEV世界坐标: X右(横向), Y前(纵向)
        """
        fx = self.camera_matrix['fx']
        fy = self.camera_matrix['fy']
        cx = self.camera_matrix['cx']
        cy = self.camera_matrix['cy']

        # 反投影到相机坐标系
        z_cam = depth  # 前向距离
        x_cam = (cx_2d - cx) * z_cam / fx  # 横向位置

        # BEV坐标: X=横向, Y=前向
        return x_cam, z_cam

    def _extract_reid(
            self,
            image: np.ndarray,
            bbox: List[float]
    ) -> np.ndarray:
        """提取ReID特征 (对齐Baseline Extractor)"""
        if self.reid_extractor is None:
            return np.zeros(512, dtype=np.float32)

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return np.zeros(512, dtype=np.float32)

            # Crop并resize到128×256 (Baseline尺寸)
            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256))

            # 归一化
            crop = crop.astype(np.float32) / 255.0
            crop = torch.from_numpy(crop).permute(2, 0, 1)

            # ImageNet归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            crop = (crop - mean) / std

            crop = crop.unsqueeze(0).to(self.device)
            if self.half:
                crop = crop.half()

            with torch.no_grad():
                feat = self.reid_extractor(crop)
                feat = feat.flatten().cpu().numpy().astype(np.float32)

            # 归一化特征
            feat = feat / (np.linalg.norm(feat) + 1e-6)

            # 确保512维
            if len(feat) < 512:
                feat = np.pad(feat, (0, 512 - len(feat)))
            return feat[:512]
        except Exception as e:
            return np.zeros(512, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 V42新增: 多源检测融合函数优化版
# ═══════════════════════════════════════════════════════════════════════════════════
def fuse_detection_positions(
        radar_detections: List[Detection],
        camera_detections: List[Detection] = None,
        fusion_threshold: float = 3.0,
        radar_weight: float = 0.9,  # 🔴 对齐Baseline: 0.9/0.1
        camera_weight: float = 0.1,  # 🔴 对齐Baseline: 0.1
        reid_only: bool = None,  # 🔴 V42+: 即使为True，现在也会保留高置信度相机目标
) -> Tuple[List[Detection], List[Detection], List[Detection]]:
    """
    融合雷达检测和相机检测 (优化版：提升召回率)

    Args:
        radar_detections: 热力图/雷达检测列表 (RADAR_ONLY)
        camera_detections: YOLO检测列表 (CAMERA_ONLY)
        fusion_threshold: 融合距离阈值(米)
        radar_weight: 雷达位置权重
        camera_weight: 相机位置权重
        reid_only: 特征级融合模式标记

    Returns:
        fusion_detections: 融合检测 (FUSION)
        unmatched_camera: 未匹配的相机检测 (CAMERA_ONLY)
        unmatched_radar: 未匹配的雷达检测 (RADAR_ONLY)
    """
    # 使用全局配置或参数覆盖
    if reid_only is None:
        reid_only = YOLO_REID_ONLY

    # 1. 处理无相机检测的情况
    if camera_detections is None or len(camera_detections) == 0:
        for det in radar_detections:
            det.fusion_state = FusionState.RADAR_ONLY
        return [], [], radar_detections

    # 2. 处理无雷达检测的情况
    if len(radar_detections) == 0:
        # 🔴 优化逻辑：即使是 reid_only 模式，如果 YOLO 很有把握，也不应该丢弃
        # 这能挽救那些雷达没扫到但相机看得很清楚的目标
        unmatched_camera = []
        for det in camera_detections:
            # 如果不是 reid_only 或者是高置信度目标，则保留
            if not reid_only or det.confidence > 0.55:
                det.fusion_state = FusionState.CAMERA_ONLY
                unmatched_camera.append(det)
        return [], unmatched_camera, []

    # 3. 计算距离矩阵
    from scipy.optimize import linear_sum_assignment

    n_radar = len(radar_detections)
    n_camera = len(camera_detections)
    cost_matrix = np.zeros((n_radar, n_camera))

    for i, r_det in enumerate(radar_detections):
        for j, c_det in enumerate(camera_detections):
            dist = np.linalg.norm(r_det.center - c_det.center)
            cost_matrix[i, j] = dist

    # 4. 匈牙利算法匹配
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    fusion_detections = []
    matched_radar = set()
    matched_camera = set()

    for r_idx, c_idx in zip(row_indices, col_indices):
        dist = cost_matrix[r_idx, c_idx]
        if dist < fusion_threshold:
            r_det = radar_detections[r_idx]
            c_det = camera_detections[c_idx]

            if reid_only:
                # 🔴 特征级融合模式：匹配上的目标，位置信赖雷达(DBSCAN)，身份信赖相机(YOLO)
                fused_center = r_det.center.copy()
                fusion_state = FusionState.RADAR_ONLY  # 这种组合通常被视为关联了特征的雷达目标
            else:
                # 决策级融合模式：位置加权
                fused_x = r_det.center[0] * 0.2 + c_det.center[0] * 0.8
                fused_y = r_det.center[1] * 0.95 + c_det.center[1] * 0.05

                fused_center = np.array([fused_x, fused_y])

                fusion_state = FusionState.FUSION

            # 创建融合检测
            fused_det = Detection(
                center=fused_center,
                fusion_state=fusion_state,
                confidence=max(r_det.confidence, c_det.confidence),
                feature=c_det.feature.copy(),  # 关键：获取 ReID 特征
                tlwh=c_det.tlwh.copy(),
                proj_xy=c_det.proj_xy.copy(),
                classes=c_det.classes,
                r_center=r_det.center.copy(),
                c_center=c_det.center.copy() if not reid_only else None,  # 如果是reid_only模式，通常不记录c_center以避免混淆
                depth=c_det.depth,
            )
            fusion_detections.append(fused_det)
            matched_radar.add(r_idx)
            matched_camera.add(c_idx)

    # 5. 处理未匹配的雷达检测（保留，作为纯雷达目标）
    unmatched_radar = []
    for i, det in enumerate(radar_detections):
        if i not in matched_radar:
            det.fusion_state = FusionState.RADAR_ONLY
            unmatched_radar.append(det)

    # 6. 处理未匹配的相机检测
    unmatched_camera = []
    for i, det in enumerate(camera_detections):
        if i not in matched_camera:
            # 🔴 关键优化：双重保险
            # 如果不是 reid_only 模式 -> 直接保留
            # 如果是 reid_only 模式 -> 只有置信度高 (>0.45) 的才保留
            # 这样既避免了低分误检，又找回了高分漏检
            if not reid_only or det.confidence > 0.45:
                det.fusion_state = FusionState.CAMERA_ONLY
                unmatched_camera.append(det)

    return fusion_detections, unmatched_camera, unmatched_radar


# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 V42新增: YOLO集成管理器
# ═══════════════════════════════════════════════════════════════════════════════════

class YOLOIntegrationManager:
    """
    YOLO集成管理器 - 对齐Baseline RadarRGBFusionNet2_20231128

    支持两种模式:
    1. 无分阶段训练: YOLO全程启用
    2. 有分阶段训练: Stage1禁用YOLO, Stage2启用YOLO
    """

    def __init__(
            self,
            enable_yolo: bool = True,
            enable_staged_training: bool = False,
            model_name: str = 'yolov5m',
            conf_threshold: float = 0.3,
            iou_threshold: float = 0.5,
            classes: List[int] = None,
            camera_matrix: Dict = None,
            enable_reid: bool = True,
            device: str = 'cuda',
    ):
        self.enable_yolo = enable_yolo
        self.enable_staged_training = enable_staged_training
        self.current_stage = 1 if enable_staged_training else 0
        self.classes = classes or [2]

        self.yolo_detector = None
        if enable_yolo:
            print("\n" + "=" * 60)
            print("🎯 初始化YOLO集成 (对齐Baseline)")
            print("=" * 60)
            self.yolo_detector = YOLODetector(
                model_name=model_name,
                pretrained=True,
                device=device,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                classes=self.classes,
                camera_matrix=camera_matrix,
                enable_reid=enable_reid,
            )
            print("=" * 60 + "\n")

    def set_stage(self, stage: int):
        """设置当前训练阶段"""
        self.current_stage = stage
        if stage == 1:
            print("📌 YOLO状态: ❌禁用 (Stage 1 深度预训练)")
        else:
            print("📌 YOLO状态: ✅启用 (Stage 2 联合训练)")

    def should_use_yolo(self) -> bool:
        """检查是否应使用YOLO"""
        if not self.enable_yolo or self.yolo_detector is None:
            return False
        if self.enable_staged_training and self.current_stage == 1:
            return False
        return True

    def detect(
            self,
            rgb_image: np.ndarray,
            depth_map: np.ndarray = None
    ) -> List[Detection]:
        """运行YOLO检测"""
        if not self.should_use_yolo():
            return []
        return self.yolo_detector.detect(rgb_image, depth_map)

    def process_detections(
            self,
            rgb_image: np.ndarray,
            depth_map: np.ndarray,
            heatmap_detections: List[Detection],
            fusion_threshold: float = 3.0,
    ) -> List[Detection]:
        """
        融合YOLO检测和热力图检测

        Args:
            rgb_image: RGB图像
            depth_map: SGDNet深度图
            heatmap_detections: 热力图检测列表
            fusion_threshold: 融合距离阈值

        Returns:
            融合后的检测列表
        """
        if not self.should_use_yolo():
            for det in heatmap_detections:
                det.fusion_state = FusionState.RADAR_ONLY
            return heatmap_detections

        # YOLO检测
        yolo_detections = self.yolo_detector.detect(rgb_image, depth_map)

        if len(yolo_detections) == 0:
            for det in heatmap_detections:
                det.fusion_state = FusionState.RADAR_ONLY
            return heatmap_detections

        # 融合
        fusion_dets, unmatched_cam, unmatched_radar = fuse_detection_positions(
            heatmap_detections, yolo_detections,
            fusion_threshold=fusion_threshold
        )

        # 合并所有检测
        return fusion_dets + unmatched_cam + unmatched_radar

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'yolo_enabled': self.enable_yolo,
            'staged_training': self.enable_staged_training,
            'current_stage': self.current_stage,
            'using_yolo': self.should_use_yolo(),
            'classes': self.classes,
        }


# 全局YOLO管理器实例（延迟初始化）
_yolo_manager = None


def get_yolo_manager(
        enable_staged: bool = False,
        device: str = 'cuda',
        camera_matrix: Dict = None
) -> YOLOIntegrationManager:
    """获取或创建全局YOLO管理器"""
    global _yolo_manager
    if _yolo_manager is None and ENABLE_YOLO:
        _yolo_manager = YOLOIntegrationManager(
            enable_yolo=True,
            enable_staged_training=enable_staged,
            model_name=YOLO_MODEL_NAME,
            conf_threshold=YOLO_CONF_THRESHOLD,
            iou_threshold=YOLO_IOU_THRESHOLD,
            classes=YOLO_CLASSES,
            camera_matrix=camera_matrix,
            enable_reid=YOLO_ENABLE_REID,
            device=device,
        )
    return _yolo_manager


def reset_yolo_manager():
    """重置YOLO管理器"""
    global _yolo_manager
    _yolo_manager = None


# ============================================================
# 🔴 V45: ReID外观特征提取器（对齐Baseline）
# 参考: RadarRGBFusionNet2_20231128/Detection/deep/feature_extractor.py
# ============================================================

class ReIDExtractor:
    """
    ReID外观特征提取器（对齐Baseline的OSNet实现）

    Baseline流程:
    1. YOLO检测得到bbox
    2. 从相机图像crop目标区域
    3. OSNet提取512维特征

    我们的实现:
    1. 从世界坐标映射到图像坐标
    2. 从相机图像crop目标区域
    3. OSNet（或ResNet18备选）提取512维特征
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.feature_dim = 512
        self.input_size = (128, 256)  # (width, height) 和Baseline一致
        self.model = None
        self.transform = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化（避免在import时加载模型）"""
        if self._initialized:
            return

        import torchvision.transforms as transforms

        # 图像预处理（和Baseline一致）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # 尝试加载OSNet（和Baseline一样）
        try:
            reid_path = '/home/claude/RadarRGBFusionNet2_20231128/Detection/deep/reid'
            if os.path.exists(reid_path):
                sys.path.insert(0, reid_path)
            from torchreid import models as reid_models
            self.model = reid_models.build_model(name='osnet_ain_x1_0', num_classes=1000)
            self.model.to(self.device)
            self.model.eval()
            print("✅ ReID: 加载OSNet成功 (对齐Baseline)")
        except Exception as e:
            print(f"⚠️ ReID: OSNet不可用({e})，使用ResNet18替代")
            try:
                import torchvision.models as models
                resnet = models.resnet18(pretrained=True)
                # 移除最后的全连接层，输出512维
                self.model = nn.Sequential(*list(resnet.children())[:-1])
                self.model.to(self.device)
                self.model.eval()
            except Exception as e2:
                print(f"⚠️ ReID: ResNet18也不可用({e2})，禁用外观特征")
                self.model = None

        self._initialized = True

    @torch.no_grad()
    def extract_features(self, camera_img, detections, bev_range=None):
        """
        从相机图像中提取检测目标的ReID特征（对齐Baseline）

        Args:
            camera_img: 相机图像 (H, W, 3) numpy array BGR格式
            detections: 检测列表 [(x, y, conf), ...] 世界坐标
            bev_range: BEV范围 {'x_min': -10, 'x_max': 10, 'y_min': 0, 'y_max': 50}

        Returns:
            features: list of numpy arrays，每个是512维特征向量
        """
        self._lazy_init()

        if self.model is None or len(detections) == 0:
            return [np.zeros(self.feature_dim, dtype=np.float32) for _ in detections]

        # 默认BEV范围 - 与Baseline一致
        if bev_range is None:
            bev_range = {'x_min': -10, 'x_max': 10, 'y_min': 0, 'y_max': 50}  # 🔧 V39修复

        img_h, img_w = camera_img.shape[:2]
        features = []
        crops = []
        valid_indices = []

        for i, det in enumerate(detections):
            world_x, world_y = det[0], det[1]

            # 世界坐标到图像坐标的映射（简化版）
            # X: [x_min, x_max] -> [0, img_w]
            norm_x = (world_x - bev_range['x_min']) / (bev_range['x_max'] - bev_range['x_min'])
            img_x = int(norm_x * img_w)

            # Y: [y_min, y_max] -> [img_h, 0]（远处在图像上方）
            norm_y = (world_y - bev_range['y_min']) / (bev_range['y_max'] - bev_range['y_min'])
            img_y = int((1 - norm_y) * img_h)

            # 根据距离估算bbox大小（Baseline做法：远处目标更小）
            distance_factor = max(0.2, 1.0 - norm_y)
            bbox_w = int(100 * distance_factor)
            bbox_h = int(200 * distance_factor)

            # 计算bbox
            x1 = max(0, img_x - bbox_w // 2)
            x2 = min(img_w, img_x + bbox_w // 2)
            y1 = max(0, img_y - bbox_h)
            y2 = min(img_h, img_y)

            # 确保bbox有效
            if x2 > x1 + 10 and y2 > y1 + 10:
                crop = camera_img[y1:y2, x1:x2]
                # Resize到标准尺寸
                crop_resized = cv2.resize(crop, self.input_size)
                crops.append(crop_resized)
                valid_indices.append(i)

        # 批量提取特征
        if len(crops) > 0:
            # 预处理
            batch = []
            for crop in crops:
                # BGR -> RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                # 归一化
                crop_tensor = self.transform(crop_rgb.astype(np.float32) / 255.0)
                batch.append(crop_tensor)

            batch_tensor = torch.stack(batch).to(self.device)

            # 提取特征
            batch_features = self.model(batch_tensor)
            batch_features = batch_features.view(batch_features.size(0), -1)  # Flatten
            batch_features = batch_features.cpu().numpy()

            # 构建结果
            result = [np.zeros(self.feature_dim, dtype=np.float32) for _ in detections]
            for idx, feat in zip(valid_indices, batch_features):
                result[idx] = feat
            return result
        else:
            return [np.zeros(self.feature_dim, dtype=np.float32) for _ in detections]


# 全局ReID提取器实例
_REID_EXTRACTOR = None


def get_reid_extractor(device='cuda'):
    """获取全局ReID特征提取器"""
    global _REID_EXTRACTOR
    if _REID_EXTRACTOR is None:
        _REID_EXTRACTOR = ReIDExtractor(device=device)
    return _REID_EXTRACTOR


print("✅ ReID特征提取器模块已加载")

# ============================================================
# 原有代码继续
# ============================================================

# 添加模块路径（确保路径正确）
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, 'RadarRGBFusionNet2_20231128', 'module')
sgd_path = os.path.join(current_dir, 'SGDNet_TI')
sgd_models_path = os.path.join(current_dir, 'SGDNet_TI', 'Models')

# 确保路径存在后再添加
if os.path.exists(module_path):
    sys.path.insert(0, module_path)
if os.path.exists(sgd_path):
    sys.path.insert(0, sgd_path)
if os.path.exists(sgd_models_path):
    sys.path.insert(0, sgd_models_path)

from one_stage_model import one_stage_model

# 尝试导入SGDNet_TI，如果失败则使用替代方案
try:
    # 🔧 使用 sgd_v2 以匹配 checkpoint (model_epoch_66.pth) 的模型结构
    from sgd_v2 import SemanticDepthNet

    SGD_AVAILABLE = True
    print("✅ SGDNet_TI模块导入成功 (使用sgd_v2兼容版本)")
except Exception as e:
    print(f"⚠️ SGDNet_TI模块导入失败，将使用替代方案: {e}")
    SGD_AVAILABLE = False


    # 创建一个替代的SemanticDepthNet类
    class SemanticDepthNet(nn.Module):
        def __init__(self, max_sample_points=512):
            """替代的SemanticDepthNet（当真实SGDNet不可用时）"""
            super(SemanticDepthNet, self).__init__()
            self.dummy_conv = nn.Conv2d(3, 1, 1)
            self.max_sample_points = max_sample_points
            print(f"⚠️ 使用替代SGDNet (max_sample_points={max_sample_points})")

        def forward(self, *args):
            # 返回与输入相同尺寸的特征图
            if len(args) > 0:
                img = args[0]
                return None, None, self.dummy_conv(img), None
            return None, None, None, None


class SpatialAttention(nn.Module):
    """空间注意力模块 - 改进多模态特征融合"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 V42新增: LSS (Lift-Splat-Shoot) 相机特征BEV投影
# ═══════════════════════════════════════════════════════════════════════════════════

class LiftSplatShoot(nn.Module):
    """
    LSS (Lift-Splat-Shoot) 模块

    将2D相机特征利用深度信息投影到BEV空间

    参考论文: Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs
              by Viewing Transformations (ECCV 2020)

    工作流程:
    1. Lift: 利用深度图将2D图像特征"提升"到3D空间
    2. Splat: 将3D特征点"泼洒"到BEV网格
    3. Shoot: 对BEV特征进行编码 (可选)

    优势:
    - 几何正确: 利用深度信息做真正的视角变换
    - 与SGDNet深度预测自然结合
    - 计算量适中
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            bev_h: int = 128,  # 🔧 对齐Baseline: 128×128
            bev_w: int = 128,  # 🔧 对齐Baseline: 128×128
            x_range: Tuple[float, float] = (-10.0, 10.0),
            y_range: Tuple[float, float] = (0.0, 50.0),
            # 🔴 新增: 对齐Baseline，先resize到640×640
            target_h: int = 640,
            target_w: int = 640,
            # 原始相机图像尺寸 (用于计算内参缩放)
            orig_h: int = 510,
            orig_w: int = 960,
            # 相机内参 (默认值来自LeopardCamera0，针对960×510)
            fx: float = 990.14,
            fy: float = 988.78,
            cx: float = 479.53,
            cy: float = 249.07,
    ):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_range = x_range
        self.y_range = y_range

        # 🔴 新增: 目标图像尺寸 (对齐Baseline的640×640)
        self.target_h = target_h
        self.target_w = target_w
        self.orig_h = orig_h
        self.orig_w = orig_w

        # 原始相机内参 (针对960×510)
        self.fx_orig = fx
        self.fy_orig = fy
        self.cx_orig = cx
        self.cy_orig = cy

        # 🔴 计算缩放后的内参 (针对target_h × target_w)
        # letterbox会保持宽高比，所以需要计算实际缩放比例
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        # padding计算
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2

        # 缩放内参
        self.fx = fx * scale
        self.fy = fy * scale
        self.cx = cx * scale + pad_w  # 加上padding偏移
        self.cy = cy * scale + pad_h

        # 🔴 BUG #10修复: 更清晰的打印信息
        if orig_h == target_h and orig_w == target_w:
            print(f"🔧 LSS内参: 输入尺寸({orig_w}×{orig_h})已匹配，无需resize")
            print(f"   fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        else:
            print(f"🔧 LSS内参缩放: 原始({orig_w}×{orig_h}) → 目标({target_w}×{target_h})")
            print(f"   fx: {fx:.2f} → {self.fx:.2f}")
            print(f"   fy: {fy:.2f} → {self.fy:.2f}")
            print(f"   cx: {cx:.2f} → {self.cx:.2f}")
            print(f"   cy: {cy:.2f} → {self.cy:.2f}")

        # 特征编码器: 对输入RGB进行特征提取
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # BEV编码器: 对Splat后的BEV特征进行处理
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def _letterbox_resize(self, img: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐Baseline的letterbox预处理：保持宽高比resize+padding

        Args:
            img: (B, 3, H, W) RGB图像
            depth: (B, 1, H, W) 深度图

        Returns:
            img_resized: (B, 3, target_h, target_w)
            depth_resized: (B, 1, target_h, target_w)
        """
        B, C, H, W = img.shape

        # 🔴 BUG #10修复: 如果输入尺寸已等于目标尺寸，直接返回，不做任何处理
        if H == self.target_h and W == self.target_w:
            return img, depth

        # 计算缩放比例（保持宽高比）
        scale = min(self.target_h / H, self.target_w / W)
        new_h = int(H * scale)
        new_w = int(W * scale)

        # Resize
        img_scaled = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        depth_scaled = F.interpolate(depth, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Padding到目标尺寸
        pad_h = (self.target_h - new_h) // 2
        pad_w = (self.target_w - new_w) // 2
        pad_h2 = self.target_h - new_h - pad_h
        pad_w2 = self.target_w - new_w - pad_w

        # F.pad: (left, right, top, bottom)
        img_padded = F.pad(img_scaled, (pad_w, pad_w2, pad_h, pad_h2), mode='constant', value=0.447)  # ImageNet mean
        depth_padded = F.pad(depth_scaled, (pad_w, pad_w2, pad_h, pad_h2), mode='constant', value=0)

        return img_padded, depth_padded

    def forward(
            self,
            img: torch.Tensor,  # (B, 3, H_img, W_img) RGB图像
            depth: torch.Tensor,  # (B, 1, H_img, W_img) 深度图
    ) -> torch.Tensor:
        """
        LSS前向传播

        Args:
            img: RGB图像 (B, 3, H_img, W_img)
            depth: 深度图 (B, 1, H_img, W_img)，值为米制深度

        Returns:
            bev_feat: BEV特征图 (B, out_channels, bev_h, bev_w)
        """
        B, C, H_img, W_img = img.shape
        device = img.device

        # 🔴 Step 0: 对齐Baseline，先letterbox resize到640×640
        img, depth = self._letterbox_resize(img, depth)
        H_img, W_img = self.target_h, self.target_w

        # Step 1: 特征提取
        img_feat = self.feature_encoder(img)  # (B, 64, H_img, W_img)
        feat_channels = img_feat.shape[1]

        # Step 2: Lift - 用深度将2D特征提升到3D
        # 确保深度图尺寸匹配
        if depth.shape[2:] != img_feat.shape[2:]:
            depth = F.interpolate(depth, size=img_feat.shape[2:], mode='bilinear', align_corners=False)

        # 生成像素坐标网格
        feat_h, feat_w = img_feat.shape[2], img_feat.shape[3]

        # 计算缩放因子 (因为特征图和原图可能尺寸不同)
        scale_x = W_img / feat_w
        scale_y = H_img / feat_h

        u = torch.arange(0, feat_w, device=device).float() * scale_x  # (feat_w,)
        v = torch.arange(0, feat_h, device=device).float() * scale_y  # (feat_h,)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')  # (feat_h, feat_w)

        # 反投影到3D相机坐标系
        # X_cam = (u - cx) * Z / fx
        # Y_cam = (v - cy) * Z / fy
        # Z_cam = Z (深度)

        depth_squeezed = depth.squeeze(1)  # (B, H, W)
        if depth_squeezed.shape[1:] != (feat_h, feat_w):
            depth_squeezed = F.interpolate(depth_squeezed.unsqueeze(1),
                                           size=(feat_h, feat_w),
                                           mode='bilinear',
                                           align_corners=False).squeeze(1)

        # 扩展坐标网格到batch维度
        u_grid = u_grid.unsqueeze(0).expand(B, -1, -1)  # (B, feat_h, feat_w)
        v_grid = v_grid.unsqueeze(0).expand(B, -1, -1)  # (B, feat_h, feat_w)

        # 计算3D坐标 (相机坐标系: X右, Y下, Z前)
        Z_cam = depth_squeezed  # (B, feat_h, feat_w)
        X_cam = (u_grid - self.cx) * Z_cam / self.fx
        Y_cam = (v_grid - self.cy) * Z_cam / self.fy

        # 转换到BEV坐标系 (X右=横向, Y前=纵向)
        # 相机坐标系 Y_cam 是向下的，但在世界坐标系中我们要前向距离
        # 所以 BEV的Y = Z_cam (前向距离)
        # BEV的X = X_cam (横向距离)
        X_bev = X_cam  # (B, feat_h, feat_w)
        Y_bev = Z_cam  # (B, feat_h, feat_w) - 深度就是前向距离

        # Step 3: Splat - 将3D特征投影到BEV网格
        bev_feat = self._splat_to_bev(img_feat, X_bev, Y_bev, device)

        # Step 4: BEV特征编码
        bev_output = self.bev_encoder(bev_feat)  # (B, out_channels, bev_h, bev_w)

        return bev_output

    def _splat_to_bev(
            self,
            img_feat: torch.Tensor,  # (B, C, H, W)
            X_bev: torch.Tensor,  # (B, H, W) 横向坐标
            Y_bev: torch.Tensor,  # (B, H, W) 纵向坐标
            device: torch.device,
    ) -> torch.Tensor:
        """
        将特征点泼洒到BEV网格

        使用scatter操作将3D点的特征累加到对应的BEV网格位置
        """
        B, C, H, W = img_feat.shape

        # 计算BEV网格坐标
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        # 归一化坐标到[0, 1]
        X_norm = (X_bev - x_min) / (x_max - x_min)  # (B, H, W)
        Y_norm = (Y_bev - y_min) / (y_max - y_min)  # (B, H, W)

        # 转换到BEV像素坐标
        bev_x = (X_norm * (self.bev_w - 1)).long()  # (B, H, W)
        bev_y = ((1 - Y_norm) * (self.bev_h - 1)).long()  # (B, H, W) - 翻转Y轴，远处在上方

        # 创建有效点mask
        valid_mask = (
                (X_norm >= 0) & (X_norm < 1) &
                (Y_norm >= 0) & (Y_norm < 1) &
                (X_bev.abs() > 0.01)  # 排除深度为0的点
        )  # (B, H, W)

        # 初始化BEV特征图
        bev_feat = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)
        bev_count = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)

        # 对每个batch进行scatter
        for b in range(B):
            mask_b = valid_mask[b]  # (H, W)
            if mask_b.sum() == 0:
                continue

            # 获取有效点的特征和坐标
            feat_b = img_feat[b]  # (C, H, W)
            feat_valid = feat_b[:, mask_b]  # (C, N_valid)

            bev_x_valid = bev_x[b][mask_b]  # (N_valid,)
            bev_y_valid = bev_y[b][mask_b]  # (N_valid,)

            # 裁剪到有效范围
            bev_x_valid = bev_x_valid.clamp(0, self.bev_w - 1)
            bev_y_valid = bev_y_valid.clamp(0, self.bev_h - 1)

            # 计算线性索引
            linear_idx = bev_y_valid * self.bev_w + bev_x_valid  # (N_valid,)

            # 使用scatter_add累加特征
            for c in range(C):
                bev_feat[b, c].view(-1).scatter_add_(
                    0, linear_idx, feat_valid[c]
                )

            # 累加计数
            bev_count[b, 0].view(-1).scatter_add_(
                0, linear_idx, torch.ones_like(linear_idx, dtype=torch.float)
            )

        # 平均化 (避免除零)
        bev_count = bev_count.clamp(min=1)
        bev_feat = bev_feat / bev_count

        return bev_feat


# ============================================================
# 🔧 2025-12-04 新增：相机视角深度图生成函数
# 与 Baseline RadarRGBFusionNet2/vis.py projection() 一致
# ============================================================

# 默认相机参数（从 LeopardCamera0.json）
# 实际图像尺寸: 960×510 (与Baseline一致)
DEFAULT_CAMERA_INTRINSIC = np.array([
    [990.1423019264267, 0.0, 479.5298113129775],
    [0.0, 988.780683582274, 249.072061811745],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DEFAULT_CAMERA_EXTRINSIC = np.array([
    [0.9985646158842507, 0.0534044699257085, -0.004082951860174227, 0.1333936485319914],
    [-0.005396032861184197, 0.02446644086851834, -0.9996860887801672, -0.3524396548028612],
    [-0.05328781036315364, 0.9982731869900049, 0.02471949440258006, -0.2159712445341051],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)


def pointcloud_to_camera_depth(points, intrinsic=None, extrinsic=None,
                               img_height=510, img_width=960, max_depth=75.0):
    """
    将3D点云投影到相机平面生成深度图（相机视角，透视投影）

    与 Baseline RadarRGBFusionNet2/vis.py projection() 一致！

    Args:
        points: (N, 3+) 点云数据 [x, y, z, ...] 在世界/LiDAR/OCU坐标系
        intrinsic: (3, 3) 相机内参矩阵，默认使用 DEFAULT_CAMERA_INTRINSIC
        extrinsic: (4, 4) 外参矩阵（点云坐标系 → 相机坐标系），默认使用 DEFAULT_CAMERA_EXTRINSIC
        img_height: 图像高度 (默认510，实际相机图像高度)
        img_width: 图像宽度 (默认960，实际相机图像宽度)
        max_depth: 最大深度，用于归一化 (默认75m，与SGDNet_TI一致)

    Returns:
        depth_map: (H, W) 归一化深度图，值域[0, 1]

    坐标系说明:
        - 点云坐标系 (Depth/OCU): X=right, Y=front, Z=up
        - 相机坐标系: X=right, Y=down, Z=front
        - 透视投影: u = fx * X/Z + cx, v = fy * Y/Z + cy
    """
    if intrinsic is None:
        intrinsic = DEFAULT_CAMERA_INTRINSIC
    if extrinsic is None:
        extrinsic = DEFAULT_CAMERA_EXTRINSIC

    if len(points) == 0:
        return np.zeros((img_height, img_width), dtype=np.float32)

    # 提取坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    n = len(x)

    # 1. 构建齐次坐标 (4, N)
    xyz1 = np.stack((x, y, z, np.ones(n)))

    # 2. 外参变换：世界/点云坐标 → 相机坐标
    xyz_cam = np.matmul(extrinsic, xyz1)  # (4, N)

    # 3. 过滤相机后方的点 (Z_cam > 0 表示在相机前方)
    mask_front = xyz_cam[2, :] > 0.5  # 至少0.5米
    if not mask_front.any():
        return np.zeros((img_height, img_width), dtype=np.float32)

    xyz_front = xyz_cam[:3, mask_front]  # (3, M)

    # 4. 内参投影
    uvz = np.matmul(intrinsic, xyz_front)  # (3, M)

    # 5. 透视除法
    u = (uvz[0, :] / uvz[2, :]).round().astype(int)
    v = (uvz[1, :] / uvz[2, :]).round().astype(int)
    depth = uvz[2, :]  # 深度 = 相机Z坐标

    # 6. 过滤范围外的点
    mask_u = (u >= 0) & (u < img_width)
    mask_v = (v >= 0) & (v < img_height)
    mask_depth = (depth > 0) & (depth < max_depth)
    mask = mask_u & mask_v & mask_depth

    u = u[mask]
    v = v[mask]
    depth = depth[mask]

    if len(u) == 0:
        return np.zeros((img_height, img_width), dtype=np.float32)

    # 7. 创建深度图（取最近点）
    depth_map = np.zeros((img_height, img_width), dtype=np.float32)
    for i in range(len(u)):
        if depth_map[v[i], u[i]] == 0 or depth[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = depth[i]

    # 8. 归一化到 [0, 1]
    depth_map = np.clip(depth_map / max_depth, 0, 1)

    return depth_map


# ========================================
# 🔧 Pseudo-LiDAR 生成辅助函数
# ========================================

def depth_to_pseudo_pointcloud(depth_map, img_size=640):
    """
    将SGDNet输出的深度图反投影成伪点云(pseudo-LiDAR)

    Args:
        depth_map: (B, 1, H_sgd, W_sgd) SGDNet输出的深度图，值范围[0,1]
        img_size: BEV图像尺寸

    Returns:
        pseudo_points: (B, N, 5) 伪点云，每个点=(x, y, z, intensity, class)
            - xyz: 3D坐标（米）
            - intensity: 置信度(固定0.5)
            - class: 类别(固定0=伪点)
    """
    import time
    t_start = time.time()  # 🔍 调试: 开始计时

    B = depth_map.shape[0]
    H_sgd, W_sgd = depth_map.shape[2], depth_map.shape[3]

    # 🔧 V46修复: 坐标范围与Baseline保持一致
    X_MIN, X_MAX = -10.0, 10.0  # 与Baseline一致 [-10, 10]m
    Y_MIN, Y_MAX = 0.0, 50.0  # 与Baseline一致 [0, 50]m (BEV范围)
    Z_MIN, Z_MAX = -5.0, 30.0  # 扩大Z范围覆盖实际数据

    # 🚀 参考Pseudo-LiDAR论文：下采样到合理数量
    # 论文中通常使用 10k-20k 点，我们设置为最多16384个点(2^14)
    # 🔧 速度优化：可以通过环境变量动态调整点数
    MAX_POINTS = int(os.environ.get('PSEUDO_POINTS', '4096'))  # 默认4096，可通过环境变量调整
    print(f"📊 Pseudo-LiDAR最大点数: {MAX_POINTS}")

    pseudo_points_list = []

    for b in range(B):
        depth_single = depth_map[b, 0].detach().cpu().numpy()  # (H_sgd, W_sgd) - 修复gradient问题

        # 生成像素网格
        v_coords, u_coords = np.mgrid[0:H_sgd, 0:W_sgd]  # (H, W)

        # 过滤有效深度点（深度>0）
        valid_mask = depth_single > 0.01  # 阈值0.01避免噪声
        valid_u = u_coords[valid_mask]
        valid_v = v_coords[valid_mask]
        valid_depth = depth_single[valid_mask]

        if len(valid_depth) == 0:
            # 没有有效点，返回空点云
            pseudo_points_list.append(np.zeros((1, 5)))
            continue
        if len(valid_depth) > MAX_POINTS:
            # 随机采样（训练时引入随机性有助于泛化）
            sample_indices = np.random.choice(len(valid_depth), MAX_POINTS, replace=False)
            valid_u = valid_u[sample_indices]
            valid_v = valid_v[sample_indices]
            valid_depth = valid_depth[sample_indices]

        # 反归一化深度：[0,1] → [Y_MIN, Y_MAX]米
        # ⚠️ 重要：这里使用SGDNet训练时的深度范围[0, 75]米
        # SGDNet输出depth ∈ [0,1]表示实际深度[0,75]米
        # 不能使用扩大后的范围[-5,105]，否则Pseudo-LiDAR坐标会错误
        y_world = valid_depth * (Y_MAX - Y_MIN) + Y_MIN

        # 从像素坐标映射到世界坐标
        # 假设深度图的像素坐标对应BEV坐标系
        norm_u = valid_u / (W_sgd - 1)  # [0, 1]
        norm_v = valid_v / (H_sgd - 1)  # [0, 1]

        x_world = norm_u * (X_MAX - X_MIN) + X_MIN
        z_world = (1 - norm_v) * (Z_MAX - Z_MIN) + Z_MIN  # v=0在上方

        # 组装伪点云: (x, y, z, intensity, class)
        N = len(x_world)
        pseudo_points = np.zeros((N, 5))
        pseudo_points[:, 0] = x_world
        pseudo_points[:, 1] = y_world
        pseudo_points[:, 2] = z_world
        pseudo_points[:, 3] = 0.5  # 伪点置信度固定0.5
        pseudo_points[:, 4] = 0  # 类别0=伪点

        pseudo_points_list.append(pseudo_points)

    # 转换为tensor并padding到相同长度
    # 🔧 修复: 限制max_len避免内存浪费（原来可能padding到1228800，现在限制为MAX_POINTS*2）
    actual_max_len = max([pts.shape[0] for pts in pseudo_points_list])
    max_len = min(MAX_POINTS * 2, actual_max_len)  # 最多32768个点（16384*2）

    pseudo_points_padded = np.zeros((B, max_len, 5))
    for b in range(B):
        L = min(pseudo_points_list[b].shape[0], max_len)  # 防止超出max_len
        pseudo_points_padded[b, :L, :] = pseudo_points_list[b][:L]

    t_end = time.time()  # 🔍 调试: 结束计时
    print(f"⏱️  [位置1-伪点云生成] 耗时: {t_end - t_start:.3f}秒")

    return torch.from_numpy(pseudo_points_padded).float()


# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 DBSCAN雷达点云聚类检测（对齐Baseline OCULiiProcess/radar_process.py）
# ═══════════════════════════════════════════════════════════════════════════════════

def get_dbscan_detections(
        raw_points: np.ndarray,
        eps: float = 0.9,
        min_samples: int = 7,
        x_range: Tuple[float, float] = (-10.0, 10.0),
        y_range: Tuple[float, float] = (0.0, 50.0),
        use_coord_swap: bool = None,
) -> List[Detection]:
    """
    对雷达点云进行DBSCAN聚类，生成检测结果（对齐Baseline）

    参考: RadarRGBFusionNet2_20231128/OCULiiProcess/radar_process.py
    - get_dbscan_points()
    - get_radar_features()

    Args:
        raw_points: (N, 5) numpy array，雷达点云 [x, y, z, velocity, SNR]
                   Baseline会重排列坐标: [x, y, z] -> [x, z, y]
        eps: DBSCAN邻域半径 (Baseline=0.9)
        min_samples: 最小样本数 (Baseline=7)
        x_range: X轴有效范围 (横向)
        y_range: Y轴有效范围 (前向)
        use_coord_swap: 是否交换Y/Z坐标，None时使用全局设置

    Returns:
        detections: Detection列表，每个聚类一个Detection
                   - center: 聚类中心 (x, y)
                   - fusion_state: RADAR_ONLY (2)
                   - r_center: 雷达中心坐标
                   - confidence: 基于点数的置信度
                   - feature: 10维雷达特征 (a1-a10)
    """
    if raw_points is None or len(raw_points) == 0:
        return []

    # 确保是numpy数组
    if isinstance(raw_points, torch.Tensor):
        raw_points = raw_points.cpu().numpy()

    # 过滤无效点（全零点）
    valid_mask = np.abs(raw_points[:, 0]) > 0.01
    points = raw_points[valid_mask].copy()

    if len(points) < min_samples:
        return []

    # 🔴 对齐Baseline: 坐标重排 [x, y, z, vel, SNR] -> [x, z, y, vel, SNR]
    # Baseline中: raw [x,y,z] 的 y=高度, z=前向 → 重排后 column1=前向, column2=高度
    if use_coord_swap is None:
        use_coord_swap = USE_COORD_SWAP

    if use_coord_swap and points.shape[1] >= 3:
        # 交换 Y 和 Z 列
        points[:, 1], points[:, 2] = points[:, 2].copy(), points[:, 1].copy()

    # DBSCAN聚类（基于XY坐标，与Baseline一致）
    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
        labels = clustering.labels_
    except Exception as e:
        print(f"⚠️ DBSCAN聚类失败: {e}")
        return []

    # 统计聚类数量（排除噪声点label=-1）
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # 🔍 调试: DBSCAN聚类结果
    noise_count = np.sum(labels == -1)
    DEBUG_DBSCAN = os.environ.get('DEBUG_DBSCAN', '0') == '1'
    if DEBUG_DBSCAN:
        print(f"🔍 [DBSCAN调试] eps={eps}, min_samples={min_samples}")
        print(f"   输入点数: {len(points)}, 聚类数: {num_clusters}, 噪声点: {noise_count}")

    if num_clusters == 0:
        return []

    detections = []

    for cluster_id in range(num_clusters):
        # 获取当前聚类的所有点
        cluster_mask = labels == cluster_id
        cluster_points = points[cluster_mask]

        if len(cluster_points) == 0:
            continue

        # 计算聚类中心
        r_center = np.mean(cluster_points[:, :2], axis=0)

        # 过滤超出范围的聚类（与Baseline一致）
        if (r_center[0] < x_range[0] or r_center[0] > x_range[1] or
                r_center[1] < y_range[0] or r_center[1] > y_range[1]):
            continue

        # 提取10维雷达特征（与Baseline完全一致）
        # a1: 点数
        a1 = cluster_points.shape[0]
        # a2: X范围
        a2 = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
        # a3: Y范围
        a3 = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
        # a4: 面积
        a4 = a2 * a3
        # a5: 密度
        a5 = a1 / (a4 + 1e-6)
        # a6: Z范围
        a6 = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2]) if cluster_points.shape[1] > 2 else 0.0
        # a7: 平均高度
        a7 = np.mean(cluster_points[:, 2]) if cluster_points.shape[1] > 2 else 0.0
        # a8: 平均速度
        a8 = np.mean(cluster_points[:, 3]) if cluster_points.shape[1] > 3 else 0.0
        # a9: 速度范围
        a9 = np.max(cluster_points[:, 3]) - np.min(cluster_points[:, 3]) if cluster_points.shape[1] > 3 else 0.0
        # a10: 最大SNR
        a10 = np.max(cluster_points[:, 4]) if cluster_points.shape[1] > 4 else 0.0

        radar_feature = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10], dtype=np.float32)

        # 基于点数计算置信度（点数越多越可信）
        confidence = min(1.0, a1 / 20.0)  # 20个点以上置信度为1.0

        # 创建Detection对象
        det = Detection(
            center=r_center.astype(np.float32),
            fusion_state=FusionState.RADAR_ONLY,
            confidence=confidence,
            feature=np.pad(radar_feature, (0, 512 - 10)),  # 填充到512维
            r_center=r_center.astype(np.float32),
        )
        detections.append(det)

    return detections


def get_dbscan_detections_batch(
        radar_points_batch: torch.Tensor,
        eps: float = 0.9,
        min_samples: int = 7,
        x_range: Tuple[float, float] = (-10.0, 10.0),
        y_range: Tuple[float, float] = (0.0, 50.0),
        use_coord_swap: bool = None,
) -> List[List[Detection]]:
    """
    批量处理雷达点云的DBSCAN聚类检测

    Args:
        radar_points_batch: (B, N, C) tensor，批量雷达点云
        eps: DBSCAN邻域半径 (Baseline=0.9)
        min_samples: 最小样本数 (Baseline=7)
        x_range: X轴有效范围
        y_range: Y轴有效范围
        use_coord_swap: 是否交换Y/Z坐标，None时使用全局设置

    Returns:
        batch_detections: List[List[Detection]]，每个样本的检测列表
    """
    if radar_points_batch is None:
        return []

    B = radar_points_batch.shape[0]
    batch_detections = []

    for b in range(B):
        points = radar_points_batch[b].cpu().numpy() if isinstance(radar_points_batch, torch.Tensor) else \
        radar_points_batch[b]
        detections = get_dbscan_detections(points, eps, min_samples, x_range, y_range, use_coord_swap)
        batch_detections.append(detections)

    return batch_detections


# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 DBSCAN雷达点云去噪（借鉴Baseline）
# ═══════════════════════════════════════════════════════════════════════════════════
def dbscan_denoise_radar(points, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, use_coord_swap=None):
    """
    使用DBSCAN对雷达点云进行去噪

    只对毫米波雷达做DBSCAN，伪点云不做（符合学术界做法）

    Args:
        points: (N, C) numpy array，点云数据 [x, y, z, ...]
        eps: DBSCAN邻域半径 (Baseline=0.9)
        min_samples: 最小样本数 (Baseline=7)
        use_coord_swap: 是否交换Y/Z坐标，None时使用全局设置

    Returns:
        filtered_points: (M, C) 去噪后的点云
    """
    if len(points) < min_samples:
        return points

    # 🔴 对齐Baseline: 坐标重排用于DBSCAN
    if use_coord_swap is None:
        use_coord_swap = USE_COORD_SWAP

    points_for_clustering = points.copy()
    if use_coord_swap and points_for_clustering.shape[1] >= 3:
        # 交换 Y 和 Z 列用于聚类
        points_for_clustering[:, 1], points_for_clustering[:, 2] = \
            points_for_clustering[:, 2].copy(), points_for_clustering[:, 1].copy()

    # 使用XYZ坐标进行聚类
    xyz = points_for_clustering[:, :3]

    # DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_

    # 过滤噪声点（label == -1 表示噪声）
    valid_mask = labels >= 0

    if not valid_mask.any():
        # 如果全是噪声，返回原始点云
        return points

    return points[valid_mask]


def preprocess_radar_dbscan(radar_points, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, use_coord_swap=None):
    """
    对批量雷达点云进行DBSCAN预处理

    Args:
        radar_points: (B, N, C) tensor，雷达点云
        eps: DBSCAN邻域半径 (Baseline=0.9)
        min_samples: 最小样本数 (Baseline=7)
        use_coord_swap: 是否交换Y/Z坐标，None时使用全局设置

    Returns:
        processed_points: (B, N, C) tensor，处理后的点云（可能有padding）
    """
    if not USE_DBSCAN:
        return radar_points

    # 🔴 修复: 检查输入形状，处理非标准输入
    if radar_points is None:
        return None

    # 确保是tensor
    if not isinstance(radar_points, torch.Tensor):
        radar_points = torch.from_numpy(np.array(radar_points)).float()

    # 检查维度
    if radar_points.dim() == 2:
        # (N, C) -> (1, N, C)
        radar_points = radar_points.unsqueeze(0)
    elif radar_points.dim() != 3:
        print(f"⚠️ preprocess_radar_dbscan: 输入形状异常 {radar_points.shape}，跳过DBSCAN")
        return radar_points

    B, N, C = radar_points.shape
    device = radar_points.device

    processed_list = []
    max_points = 0

    for b in range(B):
        pts = radar_points[b].cpu().numpy()

        # 过滤零点
        valid_mask = np.abs(pts[:, 0]) > 0.01
        valid_pts = pts[valid_mask]

        if len(valid_pts) > 0:
            # DBSCAN去噪（传入坐标重排参数）
            filtered_pts = dbscan_denoise_radar(valid_pts, eps, min_samples, use_coord_swap)
            processed_list.append(filtered_pts)
            max_points = max(max_points, len(filtered_pts))
        else:
            processed_list.append(np.zeros((0, C), dtype=np.float32))

    # Padding到相同长度
    if max_points == 0:
        max_points = N

    result = np.zeros((B, max_points, C), dtype=np.float32)
    for b, pts in enumerate(processed_list):
        if len(pts) > 0:
            n = min(len(pts), max_points)
            result[b, :n] = pts[:n]

    return torch.from_numpy(result).to(device)


# ═══════════════════════════════════════════════════════════════════════════════════
# 🔴 V42+: 特征级DBSCAN后处理 - 对融合热力图进行聚类检测
# ═══════════════════════════════════════════════════════════════════════════════════

def feature_level_dbscan_detection(
        heatmap: torch.Tensor,
        conf_threshold: float = 0.3,
        eps: float = 5.0,  # BEV像素空间的邻域半径
        min_samples: int = 3,
        x_range: Tuple[float, float] = (-10.0, 10.0),
        y_range: Tuple[float, float] = (0.0, 50.0),
) -> List[List[Tuple[float, float, float]]]:
    """
    🔴 特征级DBSCAN检测 - 对网络输出的融合热力图进行聚类

    工作流程:
    1. 从热力图中提取高置信度点 (conf > threshold)
    2. 在BEV像素空间对这些点进行DBSCAN聚类
    3. 计算每个聚类的中心作为检测位置
    4. 将像素坐标转换回世界坐标

    Args:
        heatmap: (B, 1, H, W) 网络输出的融合热力图，值域[0,1]
        conf_threshold: 置信度阈值，只聚类高于此阈值的点
        eps: DBSCAN邻域半径（像素单位）
        min_samples: DBSCAN最小样本数
        x_range: 世界坐标X范围（横向）
        y_range: 世界坐标Y范围（前向）

    Returns:
        batch_detections: List[List[(x, y, conf)]]
            每个batch的检测列表，每个检测为(世界坐标x, 世界坐标y, 置信度)
    """
    if heatmap is None:
        return []

    # 确保是numpy
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.detach().cpu().numpy()
    else:
        heatmap_np = heatmap

    B = heatmap_np.shape[0]
    H, W = heatmap_np.shape[2], heatmap_np.shape[3]

    batch_detections = []

    for b in range(B):
        # 获取单个样本的热力图
        hm = heatmap_np[b, 0]  # (H, W)

        # 应用sigmoid（如果还没有）
        if hm.max() > 1.0 or hm.min() < 0.0:
            hm = 1.0 / (1.0 + np.exp(-hm))

        # 提取高置信度点
        high_conf_mask = hm > conf_threshold
        high_conf_points = np.argwhere(high_conf_mask)  # (N, 2) - [row, col] = [y, x]

        if len(high_conf_points) < min_samples:
            batch_detections.append([])
            continue

        # 获取对应的置信度值
        conf_values = hm[high_conf_mask]

        # DBSCAN聚类（在像素空间）
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(high_conf_points)
            labels = clustering.labels_
        except Exception as e:
            batch_detections.append([])
            continue

        # 统计聚类
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if num_clusters == 0:
            batch_detections.append([])
            continue

        detections = []
        for cluster_id in range(max(labels) + 1):
            cluster_mask = labels == cluster_id
            cluster_points = high_conf_points[cluster_mask]  # (M, 2) [row, col]
            cluster_confs = conf_values[cluster_mask]

            if len(cluster_points) == 0:
                continue

            # 加权平均计算聚类中心（按置信度加权）
            weights = cluster_confs / (cluster_confs.sum() + 1e-6)
            center_row = np.sum(cluster_points[:, 0] * weights)
            center_col = np.sum(cluster_points[:, 1] * weights)
            max_conf = cluster_confs.max()

            # 像素坐标 → 世界坐标
            # col (x方向): [0, W-1] → [x_min, x_max]
            # row (y方向): [0, H-1] → [y_max, y_min] (GT热力图生成时Y翻转，解码需要逆运算)
            # 🔴 BUG修复(2025-12-21): Y轴需要翻转！
            # GT生成: center_y = (1 - norm_y) * (H-1)  → 远处(y大)在图像上方(row小)
            # 解码:   world_y = (1 - row/(H-1)) * range + min  → 逆运算恢复正确坐标
            # 验证: GT(27.23m)→row=46, 无翻转解码→17.68m(误差9.55m)❌
            #                         有翻转解码→27.32m(误差0.09m)✅
            norm_x = center_col / (W - 1)
            norm_y = 1 - center_row / (H - 1)  # 🔴 Y轴翻转：GT生成的逆运算

            world_x = norm_x * (x_range[1] - x_range[0]) + x_range[0]
            world_y = norm_y * (y_range[1] - y_range[0]) + y_range[0]

            detections.append((float(world_x), float(world_y), float(max_conf)))

        batch_detections.append(detections)

    return batch_detections


def feature_level_dbscan_nms(
        heatmap: torch.Tensor,
        conf_threshold: float = 0.3,
        nms_kernel: int = 3,
        eps: float = 5.0,
        min_samples: int = 1,  # 🔴 NMS后点稀疏，降低min_samples
        x_range: Tuple[float, float] = (-10.0, 10.0),
        y_range: Tuple[float, float] = (0.0, 50.0),
) -> List[List[Tuple[float, float, float]]]:
    """
    🔴 特征级DBSCAN检测（带NMS预处理）

    先对热力图进行maxpool NMS去除非极大值点，再进行DBSCAN聚类
    注意：NMS后点非常稀疏，每个峰值只保留一个点，所以min_samples应该设为1

    Args:
        heatmap: (B, 1, H, W) 网络输出的融合热力图
        conf_threshold: 置信度阈值
        nms_kernel: NMS池化核大小
        eps: DBSCAN邻域半径（像素单位，用于合并相邻峰值）
        min_samples: DBSCAN最小样本数（NMS后建议设为1）
        x_range: 世界坐标X范围
        y_range: 世界坐标Y范围

    Returns:
        batch_detections: List[List[(x, y, conf)]]
    """
    if heatmap is None:
        return []

    # NMS: 只保留局部极大值
    hm_sigmoid = torch.sigmoid(heatmap) if heatmap.max() > 1.0 else heatmap
    hm_max = F.max_pool2d(hm_sigmoid, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
    hm_nms = hm_sigmoid * (hm_sigmoid == hm_max).float()

    return feature_level_dbscan_detection(
        hm_nms, conf_threshold, eps, min_samples, x_range, y_range
    )


print("✅ 特征级DBSCAN后处理模块已加载")


def extract_detections_dbscan(
        heatmap: torch.Tensor,
        threshold: float = 0.3,
        bev_range: dict = None,
        nms_kernel: int = 5,
        eps: float = 8.0,
        min_samples: int = 1,
        max_detections: int = 100,
) -> List[List['Detection']]:
    """
    🔴 V42+: 使用特征级DBSCAN从热力图提取检测结果

    与 extract_detections_from_heatmap 接口兼容，可直接替换

    Args:
        heatmap: (B, 1, H, W) 检测热力图
        threshold: 检测置信度阈值
        bev_range: BEV范围 {'x_min', 'x_max', 'y_min', 'y_max'}
        nms_kernel: NMS池化核大小
        eps: DBSCAN邻域半径（像素单位）
        min_samples: DBSCAN最小样本数
        max_detections: 最大检测数量

    Returns:
        detections_list: List[List[Detection]]，每个batch的Detection列表
    """
    if bev_range is None:
        bev_range = {'x_min': -10, 'x_max': 10, 'y_min': 0, 'y_max': 50}

    x_range = (bev_range['x_min'], bev_range['x_max'])
    y_range = (bev_range['y_min'], bev_range['y_max'])

    # 调用特征级DBSCAN
    batch_results = feature_level_dbscan_nms(
        heatmap,
        conf_threshold=threshold,
        nms_kernel=nms_kernel,
        eps=eps,
        min_samples=min_samples,
        x_range=x_range,
        y_range=y_range,
    )

    # 转换为Detection对象列表
    detections_list = []
    for b, dets in enumerate(batch_results):
        batch_dets = []
        # 按置信度排序并限制数量
        sorted_dets = sorted(dets, key=lambda x: x[2], reverse=True)[:max_detections]

        for x, y, conf in sorted_dets:
            det = Detection(
                center=np.array([x, y], dtype=np.float32),
                fusion_state=FusionState.RADAR_ONLY,  # 热力图检测
                confidence=conf,
                feature=np.zeros(512, dtype=np.float32),
            )
            batch_dets.append(det)

        detections_list.append(batch_dets)

    return detections_list


# 🔴 便捷别名，与现有代码兼容
extract_detections_from_heatmap_dbscan = extract_detections_dbscan


def fuse_pointclouds(pseudo_points, radar_points, use_dbscan=True):
    """
    融合伪点云和毫米波雷达点云

    Args:
        pseudo_points: (B, N1, 5) 伪点云
        radar_points: (B, N2, 5) 毫米波雷达点云
        use_dbscan: 是否对雷达点云使用DBSCAN去噪

    Returns:
        fused_points: (B, N1+N2, 5) 融合后的点云
    """
    # 🔴 修复: 处理radar_points为None或形状异常的情况
    if radar_points is None:
        return pseudo_points

    # 确保pseudo_points是tensor
    if not isinstance(pseudo_points, torch.Tensor):
        pseudo_points = torch.from_numpy(np.array(pseudo_points)).float()

    # 确保radar_points是tensor
    if not isinstance(radar_points, torch.Tensor):
        radar_points = torch.from_numpy(np.array(radar_points)).float()

    # 🔴 修复: 检查radar_points维度
    if radar_points.dim() == 2:
        # (N, C) -> (B, N, C)，B与pseudo_points一致
        B = pseudo_points.shape[0] if pseudo_points.dim() == 3 else 1
        radar_points = radar_points.unsqueeze(0).expand(B, -1, -1)
    elif radar_points.dim() != 3:
        print(f"⚠️ fuse_pointclouds: radar_points形状异常 {radar_points.shape}，只返回伪点云")
        return pseudo_points

    # 确保设备一致
    device = pseudo_points.device
    radar_points = radar_points.to(device)

    # 🔴 对雷达点云进行DBSCAN去噪
    if use_dbscan and USE_DBSCAN:
        radar_points = preprocess_radar_dbscan(radar_points)
        if radar_points is None:
            return pseudo_points

    # 确保通道数一致
    if pseudo_points.shape[2] != radar_points.shape[2]:
        # 扩展到相同通道数
        C_max = max(pseudo_points.shape[2], radar_points.shape[2])
        if pseudo_points.shape[2] < C_max:
            padding = torch.zeros(pseudo_points.shape[0], pseudo_points.shape[1],
                                  C_max - pseudo_points.shape[2], device=device)
            pseudo_points = torch.cat([pseudo_points, padding], dim=2)
        if radar_points.shape[2] < C_max:
            padding = torch.zeros(radar_points.shape[0], radar_points.shape[1],
                                  C_max - radar_points.shape[2], device=device)
            radar_points = torch.cat([radar_points, padding], dim=2)

    # 直接拼接两个点云
    fused = torch.cat([pseudo_points, radar_points], dim=1)
    return fused


def pointcloud_to_bev(points, img_size=640):
    """
    将融合后的密集点云投影成BEV图像(5通道)

    Args:
        points: (B, N, 5) 点云数据 (x, y, z, intensity, class)
        img_size: BEV图像尺寸

    Returns:
        bev: (B, 5, H, W) BEV图像
            通道0: 密度（点数）
            通道1: 平均高度
            通道2: 最大高度
            通道3: 平均强度
            通道4: 点数（归一化）
    """
    B = points.shape[0]
    H, W = img_size, img_size

    # 🔧 V46修复: 坐标范围与train_with_kalman45.py保持一致
    X_MIN, X_MAX = -10.0, 10.0  # 🔧 V39修复: 与Baseline一致 [-10, 10]m
    # 注意: BEV范围保持[0,50]m与Baseline一致
    # 虽然depth_to_pseudo_pointcloud输出的点可能在[0,75]m范围
    # 但超出50m的点会被norm_y的clip操作截断（投射到边缘）
    Y_MIN, Y_MAX = 0.0, 50.0  # 🔧 V39修复: 与Baseline一致 [0, 50]m
    Z_MIN, Z_MAX = -5.0, 30.0  # 🔧 修复：扩大Z范围覆盖实际数据[-4.07, 24.96]m

    bev = np.zeros((B, 5, H, W), dtype=np.float32)

    for b in range(B):
        pts = points[b].cpu().numpy()  # (N, 5)

        # 过滤有效点（非padding）
        valid_mask = np.abs(pts[:, 0]) > 0.01
        pts_valid = pts[valid_mask]

        if len(pts_valid) == 0:
            continue

        x, y, z, intensity = pts_valid[:, 0], pts_valid[:, 1], pts_valid[:, 2], pts_valid[:, 3]

        # 转换到像素坐标
        # 🔴 BUG #13修复: Y轴必须翻转！对齐Baseline坐标系
        norm_x = np.clip((x - X_MIN) / (X_MAX - X_MIN), 0, 1)
        norm_y = np.clip((y - Y_MIN) / (Y_MAX - Y_MIN), 0, 1)

        pix_x = (norm_x * (W - 1)).astype(int)
        pix_y = ((1 - norm_y) * (H - 1)).astype(int)  # 🔴 Y轴翻转! 远处在上方

        # 统计每个像素的点
        for i in range(len(pix_x)):
            px, py = pix_x[i], pix_y[i]
            if 0 <= px < W and 0 <= py < H:
                bev[b, 0, py, px] += 1  # 密度
                bev[b, 1, py, px] += z[i]  # 累加高度
                bev[b, 2, py, px] = max(bev[b, 2, py, px], z[i])  # 最大高度
                bev[b, 3, py, px] += intensity[i]  # 累加强度

        # 归一化
        count = bev[b, 0]
        count_safe = np.maximum(count, 1)

        bev[b, 1] = bev[b, 1] / count_safe  # 平均高度
        bev[b, 3] = bev[b, 3] / count_safe  # 平均强度
        bev[b, 4] = np.clip(count / 10.0, 0, 1)  # 归一化点数

        # 高度归一化到[0,1]
        bev[b, 1] = np.clip((bev[b, 1] - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)
        bev[b, 2] = np.clip((bev[b, 2] - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)

    return torch.from_numpy(bev).float()


class FusionNet(nn.Module):
    """
    融合网络：结合one_stage_model + SGDNet + Pseudo-LiDAR

    融合方式：
    1. SGDNet输出伪深度图 → 反投影成伪点云
    2. 伪点云 + 毫米波点云 融合
    3. 密集点云 → BEV图像
    4. 多路融合后的特征检测
    """

    def __init__(self, pretrained=False, num_classes=1):
        super(FusionNet, self).__init__()

        # 主干网络：one_stage_model（不接受参数）
        self.one_stage = one_stage_model()

        # SGDNet 深度补全网络（如果可用）
        if SGD_AVAILABLE:
            self.sgd = SemanticDepthNet()
            print("✅ 使用真实SGDNet_TI进行深度补全")
        else:
            self.sgd = SemanticDepthNet()
            print("⚠️ 使用替代SGDNet（降级模式）")

        # 🔧 2025-12-20: LSS (Lift-Splat-Shoot) 相机特征BEV投影
        # 利用SGDNet深度预测，将相机RGB几何正确地投影到BEV空间
        #
        # 🔴 BUG #10修复: 内参必须与实际输入尺寸匹配！
        # 原始相机: 960×510 (LeopardCamera0.json)
        # DataLoader已resize到: 960×256 (SGDNet期望尺寸)
        # 因此LSS不需要再做letterbox，直接用960×256的内参
        #
        # 内参缩放计算:
        # - Width: 960→960, scale_w = 1.0
        # - Height: 510→256, scale_h = 256/510 ≈ 0.502
        # - fx' = 990.14 * 1.0 = 990.14
        # - fy' = 988.78 * 0.502 = 496.37
        # - cx' = 479.53 * 1.0 = 479.53
        # - cy' = 249.07 * 0.502 = 125.03
        self.lss = LiftSplatShoot(
            in_channels=3,
            out_channels=1,
            bev_h=128,  # 🔧 对齐Baseline: 128×128
            bev_w=128,  # 🔧 对齐Baseline: 128×128
            x_range=(-10.0, 10.0),  # 横向范围
            y_range=(0.0, 50.0),  # 前向范围
            # 🔴 BUG #10修复: 使用实际输入尺寸，禁用内部letterbox
            target_h=256,  # 输入已是256，不需要再resize
            target_w=960,  # 输入已是960，不需要再resize
            orig_h=256,  # 实际输入高度
            orig_w=960,  # 实际输入宽度
            # 🔴 BUG #10修复: 使用针对960×256的正确内参
            fx=990.14,  # 宽度不变，fx不变
            fy=988.78 * (256.0 / 510.0),  # 高度缩放，fy按比例缩放
            cx=479.53,  # 宽度不变，cx不变
            cy=249.07 * (256.0 / 510.0),  # 高度缩放，cy按比例缩放
        )
        print("✅ LSS相机BEV投影模块已加载 (内参已适配960×256输入)")

        # 保留camera_conv作为备用（当深度不可用时）
        self.camera_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # 特征融合层（3路融合：radar + pseudo_lidar + camera）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # CBAM注意力机制
        self.attention = CBAM(in_planes=16)

        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # 🔴 BUG #12修复: 检测头bias初始化为负值
        # ═══════════════════════════════════════════════════════════════════════════
        # 问题: PyTorch默认bias=0, sigmoid(0)=0.5, 导致所有像素初始概率为50%
        # 解决: 将最后一层bias初始化为-4, sigmoid(-4)≈0.018, 接近稀疏GT分布
        # 参考: CenterNet, FCOS等检测器的标准做法
        self._init_detection_head_bias()

    def _init_detection_head_bias(self):
        """初始化检测头的bias为负值，使sigmoid输出初始接近0"""
        INIT_BIAS = -4.0  # sigmoid(-4) ≈ 0.018, 接近稀疏GT的正样本比例
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                if m.bias is not None:
                    nn.init.constant_(m.bias, INIT_BIAS)
                    print(
                        f"✅ BUG #12修复: 检测头bias初始化为{INIT_BIAS}, sigmoid≈{torch.sigmoid(torch.tensor(INIT_BIAS)).item():.4f}")

    def forward(self, dynamic_HM, static_HM, input_img=None, input_radar=None,
                input_velo=None, input_knn=None, segmap=None,
                radar_points=None):
        """
        前向传播

        Args:
            dynamic_HM: (B, 3, H, W) 动态热力图
            static_HM: (B, 3, H, W) 静态热力图
            input_img: (B, 3, H_img, W_img) RGB图像（用于SGDNet和相机特征提取）
            input_radar: (B, H_img, W_img) 稀疏雷达深度图（用于SGDNet）
            input_velo: (B, H_img, W_img) LiDAR深度图（用于SGDNet监督）
            input_knn: (B, 6, H_img, W_img) KNN特征
            segmap: (B, 6, H_img, W_img) 语义分割图
            radar_points: (B, N, 5) 原始雷达点云（可选，用于融合）

        Returns:
            final_output: (B, 1, H, W) 检测热力图
            depth_pred: (B, 1, H_sgd, W_sgd) 深度预测（用于监督）

        三路融合架构:
            1. radar_feat: one_stage_model处理动静态热力图 (B,1,H,W)
            2. pseudo_feat: SGDNet深度→伪点云→BEV (B,1,H,W)
            3. camera_feat: 相机RGB图像特征 (B,1,H,W)
        """

        # one_stage_model 处理动态和静态热力图
        radar_feat = self.one_stage(dynamic_HM, static_HM)  # (B, 1, H, W)

        # ========================================
        # 🔧 SGDNet + Pseudo-LiDAR 分支
        # ========================================

        pseudo_lidar_bev = None
        depth_pred = None

        if input_img is not None and input_radar is not None and self.sgd is not None:
            try:
                device = input_img.device

                # 确保所有输入在同一设备
                if input_velo is None:
                    input_velo = torch.zeros_like(input_radar)
                if input_knn is None:
                    B, H_img, W_img = input_radar.shape
                    input_knn = torch.zeros(B, 6, H_img, W_img, device=device)
                if segmap is None:
                    B, H_img, W_img = input_radar.shape
                    segmap = torch.zeros(B, 6, H_img, W_img, device=device)

                input_img = input_img.to(device)
                input_radar = input_radar.to(device)
                input_velo = input_velo.to(device)
                input_knn = input_knn.to(device)
                segmap = segmap.to(device)

                # 🔧 V39修复: SGDNet 输入尺寸 256×960（方案A: 直接resize）
                # SGDNet_TI 训练时使用的是 256×960，必须保持一致
                SGD_HEIGHT = 256
                SGD_WIDTH = 960  # 🔧 V39修复: 改回960，与SGDNet训练尺寸一致

                # 获取当前尺寸
                _, _, cur_H, cur_W = input_img.shape

                # 如果尺寸不匹配，进行 resize
                if cur_H != SGD_HEIGHT or cur_W != SGD_WIDTH:
                    input_img_sgd = F.interpolate(input_img, size=(SGD_HEIGHT, SGD_WIDTH), mode='bilinear',
                                                  align_corners=False)
                    input_radar_sgd = F.interpolate(input_radar.unsqueeze(1), size=(SGD_HEIGHT, SGD_WIDTH),
                                                    mode='bilinear', align_corners=False).squeeze(1)
                    input_velo_sgd = F.interpolate(input_velo.unsqueeze(1), size=(SGD_HEIGHT, SGD_WIDTH),
                                                   mode='bilinear', align_corners=False).squeeze(1)
                    input_knn_sgd = F.interpolate(input_knn, size=(SGD_HEIGHT, SGD_WIDTH), mode='bilinear',
                                                  align_corners=False)
                    segmap_sgd = F.interpolate(segmap, size=(SGD_HEIGHT, SGD_WIDTH), mode='bilinear',
                                               align_corners=False)
                else:
                    input_img_sgd = input_img
                    input_radar_sgd = input_radar
                    input_velo_sgd = input_velo
                    input_knn_sgd = input_knn
                    segmap_sgd = segmap

                # 步骤1: SGDNet生成伪深度图（使用resize后的输入）
                # 🔧 V38 调试: 打印一次输入尺寸
                if not hasattr(self, '_v38_debug_printed'):
                    self._v38_debug_printed = True
                    print(f"\n🔧 V38 SGDNet 输入尺寸: {input_img_sgd.shape[-2]}×{input_img_sgd.shape[-1]}")

                coarse_pred, cls_pred, res, radarF = self.sgd(
                    input_img_sgd, input_radar_sgd, input_velo_sgd, input_knn_sgd, segmap_sgd
                )

                # 🔧 保存深度预测（res是最终融合的深度）
                depth_pred = res

                if res is not None:
                    # 步骤2: 伪深度图 → 伪点云
                    pseudo_points = depth_to_pseudo_pointcloud(res, img_size=radar_feat.shape[-1])

                    pseudo_points = pseudo_points.to(device)

                    # 步骤3: 融合伪点云 + 毫米波点云（如果提供）
                    if radar_points is not None:
                        radar_points = radar_points.to(device)
                        fused_points = fuse_pointclouds(pseudo_points, radar_points)
                    else:
                        fused_points = pseudo_points

                    # 步骤4: 密集点云 → BEV图像
                    pseudo_lidar_bev = pointcloud_to_bev(fused_points, img_size=radar_feat.shape[-1])

                    pseudo_lidar_bev = pseudo_lidar_bev.to(device)

            except Exception as e:
                print(f"⚠️ Pseudo-LiDAR生成失败: {e}")
                import traceback
                traceback.print_exc()
                pseudo_lidar_bev = None

        # ========================================
        # 🔧 三路特征融合: radar + pseudo + camera
        # ========================================

        # 🔧 2025-12-20: 使用LSS (Lift-Splat-Shoot) 进行相机特征BEV投影
        # 利用SGDNet的深度预测，几何正确地将相机RGB投影到BEV空间
        if input_img is not None:
            bev_h, bev_w = radar_feat.shape[2], radar_feat.shape[3]

            if depth_pred is not None:
                # 🔴 使用LSS进行几何正确的BEV投影
                try:
                    # depth_pred是归一化的[0,1]深度，需要转换到米制
                    # SGDNet输出范围[0,1]对应[0, 75m]
                    MAX_DEPTH = 75.0
                    depth_meters = depth_pred * MAX_DEPTH  # (B, 1, H, W) 米制深度

                    # LSS需要和input_img尺寸一致
                    if depth_meters.shape[2:] != input_img.shape[2:]:
                        depth_meters = F.interpolate(
                            depth_meters,
                            size=input_img.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    # 调用LSS模块
                    camera_feat = self.lss(input_img, depth_meters)  # (B, 1, bev_h, bev_w)

                    # 确保尺寸与radar_feat一致
                    if camera_feat.shape[2:] != (bev_h, bev_w):
                        camera_feat = F.interpolate(
                            camera_feat,
                            size=(bev_h, bev_w),
                            mode='bilinear',
                            align_corners=False
                        )

                except Exception as e:
                    print(f"⚠️ LSS投影失败，降级使用resize: {e}")
                    # 降级：简单resize
                    camera_img_resized = F.interpolate(input_img, size=(bev_h, bev_w), mode='bilinear',
                                                       align_corners=False)
                    camera_feat = self.camera_conv(camera_img_resized)
            else:
                # 无深度预测时，降级使用简单resize（几何不正确但可用）
                camera_img_resized = F.interpolate(input_img, size=(bev_h, bev_w), mode='bilinear', align_corners=False)
                camera_feat = self.camera_conv(camera_img_resized)  # (B, 1, H, W)
        else:
            # 无相机图像时用零填充
            camera_feat = torch.zeros_like(radar_feat)

        if pseudo_lidar_bev is not None:
            # 使用Pseudo-LiDAR BEV（5通道 → 1通道压缩）
            pseudo_feat = torch.mean(pseudo_lidar_bev, dim=1, keepdim=True)  # (B, 1, H, W)

            # 三路融合: radar + pseudo-LiDAR + camera
            fusion_feat = torch.cat([radar_feat, pseudo_feat, camera_feat], dim=1)  # (B, 3, H, W)
        else:
            # 降级：pseudo失败时，用radar代替保持3通道
            fusion_feat = torch.cat([radar_feat, radar_feat, camera_feat], dim=1)  # (B, 3, H, W)

        # 特征融合和增强
        fusion_output = self.fusion_conv(fusion_feat)  # (B, 16, H, W)

        # 应用注意力机制
        attention_output = self.attention(fusion_output)  # (B, 16, H, W)

        # 最终检测头
        final_output = self.detection_head(attention_output)  # (B, 1, H, W)

        # 🔧 返回检测结果和深度预测（用于监督）
        return final_output, depth_pred

    def to(self, device):
        """Override to method to ensure SGD components are properly moved to device"""
        result = super().to(device)
        # Ensure SGD module components are properly moved to the same device
        if hasattr(self, 'sgd') and self.sgd is not None:
            try:
                self.sgd = self.sgd.to(device)
            except Exception as e:
                print(f"⚠️ SGD设备移动失败: {e}")
        return result

    def _load_erfnet_pretrained(self):
        """加载ERFNet encoder预训练权重"""
        pretrained_path = os.path.join(current_dir, 'SGDNet_TI', 'pretrained_models',
                                       'erfnet_encoder_pretrained.pth.tar')

        if not os.path.exists(pretrained_path):
            print(f"⚠️ 未找到ERFNet预训练权重: {pretrained_path}")
            print("   将从头开始训练ERFNet encoder")
            return

        try:
            print(f"🔄 加载ERFNet encoder预训练权重...")
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # 从checkpoint中提取state_dict
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
                print(f"   从checkpoint中提取state_dict (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                pretrained_dict = checkpoint

            # 获取目标模型的state_dict (self.sgd.backbone是ERFNet)
            target_state = self.sgd.backbone.state_dict()

            # 加载预训练权重到backbone (ERFNet encoder)
            loaded_count = 0
            skipped_count = 0
            skipped_layers = []  # 记录跳过的层名称

            for name, val in pretrained_dict.items():
                # 去除multi GPU前缀 "module."
                mono_name = name[7:] if name.startswith('module.') else name

                # 去除 "features." 前缀（如果存在），但保留 "encoder."
                # 预训练: module.features.encoder.XXX -> features.encoder.XXX -> encoder.XXX ✓
                if mono_name.startswith('features.'):
                    mono_name = mono_name[len('features.'):]  # 只去掉 "features."，保留 "encoder."

                # 如果key不在目标模型中，跳过
                if mono_name not in target_state:
                    skipped_count += 1
                    continue

                # 尝试复制权重
                try:
                    target_state[mono_name].copy_(val)
                    loaded_count += 1
                except RuntimeError as e:
                    # 只记录，不打印每个警告（减少输出）
                    skipped_layers.append(mono_name)
                    skipped_count += 1
                    continue

            print(f"✅ ERFNet encoder预训练权重加载成功!")
            print(f"   - 成功加载: {loaded_count} 个参数")
            print(f"   - 跳过: {skipped_count} 个参数")
            # 可选：显示跳过的层（调试用）
            if skipped_layers and len(skipped_layers) <= 10:
                print(f"   - 形状不匹配的层（预期行为，输入通道数不同）: {len(skipped_layers)}个")

        except Exception as e:
            print(f"⚠️ ERFNet预训练权重加载失败: {e}")
            print("   将从头开始训练ERFNet encoder")


if __name__ == '__main__':
    # 测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet().to(device)
    model.eval()

    B, H, W = 2, 128, 128  # 🔧 对齐Baseline: 128×128
    # one_stage_model 期望输入格式: (B, H, W, C) 不是 (B, C, H, W)
    dynamic_HM = torch.rand(B, H, W, 3).to(device)  # (B, H, W, 3)
    static_HM = torch.rand(B, H, W, 3).to(device)  # (B, H, W, 3)
    input_img = torch.rand(B, 3, H, W * 3).to(device)  # 相机RGB图像（也用于特征提取）
    input_radar = torch.rand(B, H, W * 3).to(device)
    input_velo = torch.rand(B, H, W * 3).to(device)  # Velodyne激光雷达，用于监督
    input_knn = torch.rand(B, 6, H, W * 3).to(device)
    segmap = torch.rand(B, 6, H, W * 3).to(device)

    with torch.no_grad():
        out = model(dynamic_HM, static_HM, input_img, input_radar, input_velo, input_knn, segmap)
    print("✅ 模型前向传播成功")

    # 🔧 修复: 模型现在返回 (final_output, depth_pred) 元组
    if isinstance(out, tuple):
        final_output, depth_pred = out
        print(f"   检测输出 shape: {final_output.shape}")
        print(f"   深度预测 shape: {depth_pred.shape if depth_pred is not None else 'None'}")
        print(f"   预期: (B={B}, C=1, H={H}, W={W})")
    else:
        print(f"   输出 shape: {out.shape}")
        print(f"   预期: (B={B}, C=1, H={H}, W={W})")

    # 测试相机深度图生成
    print("\n" + "=" * 60)
    print("测试 pointcloud_to_camera_depth 函数")
    print("=" * 60)

    # 创建测试点云
    test_points = np.array([
        [0, 10, 0],  # 正前方10米
        [5, 20, 0],  # 右前方
        [-5, 30, 0],  # 左前方
        [0, 50, 1],  # 远处
    ], dtype=np.float32)

    depth_map = pointcloud_to_camera_depth(test_points)
    print(f"深度图形状: {depth_map.shape}")
    print(f"非零像素数: {np.count_nonzero(depth_map)}")
    print(f"深度范围: [{depth_map.min():.4f}, {depth_map.max():.4f}]")

# ============================================================
# 🔴 V45: 卡尔曼滤波跟踪器（从train_with_kalman.py封装）
# 参考: RadarRGBFusionNet2_20231128/Track/track.py
# ============================================================

from scipy.optimize import linear_sum_assignment as hungarian_algorithm
from collections import defaultdict


class KalmanFilterFusion:
    """6维卡尔曼滤波器 - 参考Baseline
    状态: [x, y, orientation, vx, vy, v_orientation]
    观测: [x, y, orientation]
    """

    def __init__(self, dim_x=6, dim_z=3):
        self.dt = 1.0
        self.dim_x = dim_x
        self.dim_z = dim_z

        self._motion_mat = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        self.Q = np.diag([0.005, 0.005, 0.05, 0.005, 0.005, 0.05]).astype(np.float32)
        self.R = np.diag([0.005, 0.005, 0.05]).astype(np.float32)

        self.mean = np.zeros((dim_x, 1), dtype=np.float32)
        self.covariance = np.eye(dim_x, dtype=np.float32)

        self._std_weight_position = 1. / 20
        self._std_weight_position_velocity = 1.
        self._std_weight_orientation = 1. / 20
        self._std_weight_orientation_velocity = 1. / 50

    def initiate(self, x, y, orientation=0.0):
        self.mean = np.array([x, y, orientation, 0, 0, 0], dtype=np.float32).reshape(self.dim_x, 1)
        std = [
            self._std_weight_position, self._std_weight_position, self._std_weight_orientation,
            self._std_weight_position_velocity, self._std_weight_position_velocity,
            self._std_weight_orientation_velocity
        ]
        self.covariance = np.diag(np.square(std)).astype(np.float32)
        return self.mean.flatten(), self.covariance

    def predict(self):
        self.mean = self._motion_mat @ self.mean
        self.covariance = self._motion_mat @ self.covariance @ self._motion_mat.T + self.Q
        return self.mean.flatten(), self.covariance

    def update(self, measurement):
        z = np.array(measurement, dtype=np.float32).reshape(self.dim_z, 1)
        PHT = self.covariance @ self.H.T
        S = self.H @ PHT + self.R
        K = PHT @ np.linalg.inv(S)
        innovation = z - self.H @ self.mean
        self.mean = self.mean + K @ innovation
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.covariance = I_KH @ self.covariance
        return self.mean.flatten(), self.covariance


def speed_direction(pre_xy, curr_xy):
    """计算速度方向"""
    x1, y1 = pre_xy[0], pre_xy[1]
    x2, y2 = curr_xy[0], curr_xy[1]
    speed = np.array([y2 - y1, x2 - x1])
    norm = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-6
    return speed / norm


def k_previous_obs(observations, cur_age, k):
    """获取k帧前的观测"""
    if len(observations) == 0:
        return np.array([-100, -100, -1], dtype=np.float32)
    for i in range(k):
        dt = cur_age - i - 1
        if dt in observations:
            return observations[dt]
    max_age = max(observations.keys())
    return observations[max_age]


class KalmanTrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class KalmanTrack:
    """V45 轨迹 - 6维状态 + 外观特征（对齐Baseline）"""
    _next_id = 1

    def __init__(self, x, y, n_init=2, max_age=5, delta_t=3, feature=None):
        self.kf = KalmanFilterFusion()
        self.mean, self.covariance = self.kf.initiate(x, y, orientation=0.0)
        self.track_id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age
        self.delta_t = delta_t
        self.velocity = None
        self.track_orientation = 0.0
        self.last_observation = np.array([-100, -100, -1], dtype=np.float32)
        self.observations = {}
        self.state = KalmanTrackState.Confirmed if self.hits >= self.n_init else KalmanTrackState.Tentative
        # 🔴 V45: 外观特征支持（对齐Baseline）
        self.feature = feature
        self.feature_history = []

    @property
    def position(self):
        return self.mean[:2]

    @property
    def orientation(self):
        return self.mean[2] if len(self.mean) > 2 else 0.0

    def predict(self):
        self.mean, self.covariance = self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, x, y, feature=None):
        if self.last_observation[2] >= 0:
            prev_obs = k_previous_obs(self.observations, self.age, self.delta_t)
            if prev_obs[2] >= 0:
                self.velocity = speed_direction(prev_obs[:2], [x, y])
        if self.velocity is not None:
            self.track_orientation = np.arctan2(self.velocity[0], self.velocity[1])
        measurement = [x, y, self.track_orientation]
        self.mean, self.covariance = self.kf.update(measurement)
        self.last_observation = np.array([x, y, 1.0], dtype=np.float32)
        self.observations[self.age] = self.last_observation.copy()
        self.hits += 1
        self.time_since_update = 0
        if self.state == KalmanTrackState.Tentative and self.hits >= self.n_init:
            self.state = KalmanTrackState.Confirmed
        # 🔴 V45: 更新外观特征（指数移动平均）
        if feature is not None:
            if self.feature is None:
                self.feature = feature
            else:
                alpha = 0.7
                self.feature = alpha * feature + (1 - alpha) * self.feature
            self.feature_history.append(feature)
            if len(self.feature_history) > 10:
                self.feature_history.pop(0)

    def mark_missed(self):
        if self.time_since_update > self.max_age:
            self.state = KalmanTrackState.Deleted

    def is_confirmed(self):
        return self.state == KalmanTrackState.Confirmed

    def is_deleted(self):
        return self.state == KalmanTrackState.Deleted


# ============================================================
# 🔴 V45 优化版: 包含 BYTETrack两阶段匹配 + 参数调优
# ============================================================

class KalmanFilterFusion:
    """6维卡尔曼滤波器 - 参数优化版 (Opt 4: 减少滞后)"""

    def __init__(self, dim_x=6, dim_z=3):
        self.dt = 1.0
        self.dim_x = dim_x
        self.dim_z = dim_z

        self._motion_mat = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # 🔴 Optimization 4: 调整 Q 矩阵 (过程噪声)
        # 增大过程噪声，告诉滤波器"目标运动状态可能发生突变"
        # 从而减少对历史预测的过度依赖，减轻转弯或加速时的滞后
        # 原值: 0.005 -> 新值: 0.05 (位置), 0.1 (速度)
        self.Q = np.diag([0.05, 0.05, 0.1, 0.05, 0.05, 0.1]).astype(np.float32)

        # 观测噪声 R (根据实际传感器抖动调整，保持适中)
        self.R = np.diag([0.1, 0.1, 0.1]).astype(np.float32)

        self.mean = np.zeros((dim_x, 1), dtype=np.float32)
        self.covariance = np.eye(dim_x, dtype=np.float32)

        self._std_weight_position = 1. / 20
        self._std_weight_position_velocity = 1.
        self._std_weight_orientation = 1. / 20
        self._std_weight_orientation_velocity = 1. / 50

    def initiate(self, x, y, orientation=0.0):
        self.mean = np.array([x, y, orientation, 0, 0, 0], dtype=np.float32).reshape(self.dim_x, 1)
        std = [
            self._std_weight_position, self._std_weight_position, self._std_weight_orientation,
            self._std_weight_position_velocity, self._std_weight_position_velocity,
            self._std_weight_orientation_velocity
        ]
        self.covariance = np.diag(np.square(std)).astype(np.float32)
        return self.mean.flatten(), self.covariance

    def predict(self):
        self.mean = self._motion_mat @ self.mean
        self.covariance = self._motion_mat @ self.covariance @ self._motion_mat.T + self.Q
        return self.mean.flatten(), self.covariance

    def update(self, measurement):
        z = np.array(measurement, dtype=np.float32).reshape(self.dim_z, 1)
        PHT = self.covariance @ self.H.T
        S = self.H @ PHT + self.R
        K = PHT @ np.linalg.inv(S)
        innovation = z - self.H @ self.mean
        self.mean = self.mean + K @ innovation
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.covariance = I_KH @ self.covariance
        return self.mean.flatten(), self.covariance


class KalmanMOTATracker:
    """V45 跟踪器 - 包含 BYTETrack 两阶段匹配 + 外观门控"""

    DISTANCE_THRESHOLD_PRIMARY = 3
    DISTANCE_THRESHOLD_SECONDARY = 4.0  # 稍微放宽给低分匹配
    MOTA_EVAL_THRESHOLD = 6.0
    VDC_WEIGHT = 5.0

    APPEARANCE_GATE_THRESHOLD = 0.4
    APPEARANCE_WEIGHT = 2.0

    # 🔴 Optimization 5: 调整生命周期参数默认值
    # max_age: 5 -> 15 (允许更长时间的遮挡)
    # n_init: 2 -> 3 (确认轨迹需要更严格，减少FP)
    def __init__(self, max_age=15, n_init=2, max_distance=6.0):
        self.max_age = max_age
        self.n_init = n_init
        self.max_distance = max_distance

        self.tracks = []
        self.frame_count = 0

        # 统计数据
        self.total_gt = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_idsw = 0
        self.total_matches = 0
        self.match_distances = []
        self.gt_to_track_id = {}

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        KalmanTrack._next_id = 1
        self.gt_to_track_id = {}

    def predict(self):
        for track in self.tracks:
            track.predict()

    # ... (保留 _get_distance_cost_matrix, _get_orientation_cost_matrix, _get_appearance_cost_matrix 不变) ...
    def _get_distance_cost_matrix(self, detections, tracks_indices=None):
        """计算距离矩阵 (支持指定轨迹子集)"""
        # 如果未指定 indices，使用所有轨迹
        if tracks_indices is None:
            tracks_indices = list(range(len(self.tracks)))

        if len(tracks_indices) == 0 or len(detections) == 0:
            return np.zeros((len(tracks_indices), len(detections)), dtype=np.float32)

        cost_matrix = np.ones((len(tracks_indices), len(detections)), dtype=np.float32) * 1e6

        for i, t_idx in enumerate(tracks_indices):
            track = self.tracks[t_idx]
            for d, det in enumerate(detections):
                # 兼容 Detection 对象或 (x,y) 元组
                det_x = det.center[0] if hasattr(det, 'center') else det[0]
                det_y = det.center[1] if hasattr(det, 'center') else det[1]

                dist = np.sqrt((track.position[0] - det_x) ** 2 + (track.position[1] - det_y) ** 2)
                cost_matrix[i, d] = dist
        return cost_matrix

    def _get_orientation_cost_matrix(self, detections, tracks_indices=None):
        if tracks_indices is None: tracks_indices = list(range(len(self.tracks)))
        if len(tracks_indices) == 0 or len(detections) == 0:
            return np.zeros((len(tracks_indices), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(tracks_indices), len(detections)), dtype=np.float32)
        for i, t_idx in enumerate(tracks_indices):
            track = self.tracks[t_idx]
            if track.velocity is None: continue

            prev_obs = k_previous_obs(track.observations, track.age, track.delta_t)
            if prev_obs[2] < 0: continue

            for d, det in enumerate(detections):
                det_x = det.center[0] if hasattr(det, 'center') else det[0]
                det_y = det.center[1] if hasattr(det, 'center') else det[1]

                actual_dy = det_y - prev_obs[1]
                actual_dx = det_x - prev_obs[0]
                actual_norm = np.sqrt(actual_dy ** 2 + actual_dx ** 2) + 1e-6
                actual_direction = np.array([actual_dy / actual_norm, actual_dx / actual_norm])

                cos_sim = np.dot(track.velocity, actual_direction)
                cost_matrix[i, d] = (1 - cos_sim) * self.VDC_WEIGHT
        return cost_matrix

    def _get_appearance_cost_matrix(self, detection_features, tracks_indices=None):
        if tracks_indices is None: tracks_indices = list(range(len(self.tracks)))
        if len(tracks_indices) == 0 or detection_features is None or len(detection_features) == 0:
            return np.zeros((len(tracks_indices), len(detection_features) if detection_features else 0),
                            dtype=np.float32)

        cost_matrix = np.zeros((len(tracks_indices), len(detection_features)), dtype=np.float32)

        for i, t_idx in enumerate(tracks_indices):
            track = self.tracks[t_idx]
            if track.feature is None:
                cost_matrix[i, :] = 0.5
                continue

            track_feat = track.feature.flatten()
            track_norm = np.linalg.norm(track_feat) + 1e-6

            for d, det_feat in enumerate(detection_features):
                if det_feat is None:
                    cost_matrix[i, d] = 0.5
                    continue
                det_feat_flat = det_feat.flatten()
                det_norm = np.linalg.norm(det_feat_flat) + 1e-6
                cos_sim = np.dot(track_feat, det_feat_flat) / (track_norm * det_norm)
                cost_matrix[i, d] = 1.0 - cos_sim
        return cost_matrix

    def _match_cascade(self, tracks_indices, detections, detection_features=None):
        """Stage 1: 全特征级联匹配 (距离 + 方向 + 外观)"""
        if len(tracks_indices) == 0 or len(detections) == 0:
            return [], tracks_indices, list(range(len(detections)))

        distance_cost = self._get_distance_cost_matrix(detections, tracks_indices)
        orientation_cost = self._get_orientation_cost_matrix(detections, tracks_indices)
        combined_cost = distance_cost + orientation_cost

        # 外观门控
        if detection_features is not None and len(detection_features) > 0:
            appearance_cost = self._get_appearance_cost_matrix(detection_features, tracks_indices)
            gate_mask = appearance_cost > self.APPEARANCE_GATE_THRESHOLD
            combined_cost[gate_mask] = 1e6
            combined_cost += appearance_cost * self.APPEARANCE_WEIGHT

        row_indices, col_indices = hungarian_algorithm(combined_cost)

        matches = []
        unmatched_tracks = list(tracks_indices)
        unmatched_detections = list(range(len(detections)))

        for row, col in zip(row_indices, col_indices):
            t_idx = tracks_indices[row]
            d_idx = col

            final_cost = combined_cost[row, col]
            phys_dist = distance_cost[row, col]

            if final_cost < 1e5 and phys_dist < self.DISTANCE_THRESHOLD_PRIMARY:
                matches.append((t_idx, d_idx))
                if t_idx in unmatched_tracks: unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_detections: unmatched_detections.remove(d_idx)

        return matches, unmatched_tracks, unmatched_detections

    def _match_distance_only(self, tracks_indices, detections):
        """Stage 2: 仅距离匹配 (用于低分检测)"""
        if len(tracks_indices) == 0 or len(detections) == 0:
            return [], tracks_indices, list(range(len(detections)))

        # 仅使用物理距离
        cost_matrix = self._get_distance_cost_matrix(detections, tracks_indices)
        row_indices, col_indices = hungarian_algorithm(cost_matrix)

        matches = []
        unmatched_tracks = list(tracks_indices)
        unmatched_detections = list(range(len(detections)))

        for row, col in zip(row_indices, col_indices):
            t_idx = tracks_indices[row]
            d_idx = col
            dist = cost_matrix[row, col]

            # 使用稍宽松的阈值
            if dist < self.DISTANCE_THRESHOLD_SECONDARY:
                matches.append((t_idx, d_idx))
                if t_idx in unmatched_tracks: unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_detections: unmatched_detections.remove(d_idx)

        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections, gt_positions=None, gt_ids=None, detection_features=None):
        """
        🔴 Optimization 3: BYTETrack 两阶段匹配逻辑

        Args:
            detections: Detection对象列表 (含有 .confidence)
            gt_positions: 真值位置 (评估用)
            gt_ids: 真值ID (评估用)
            detection_features: 对应的特征列表
        """
        self.frame_count += 1
        self.predict()

        # 1. 准备数据: 分离高分和低分检测
        high_score_dets = []
        high_score_feats = []
        high_score_indices = []  # 记录原始索引以便映射回 gt/debug

        low_score_dets = []
        low_score_feats = []  # 低分检测通常特征不准，但为了对齐还是取出来

        CONF_THRESH = 0.5  # 高分阈值

        for i, det in enumerate(detections):
            # 获取置信度 (兼容元组或对象)
            conf = det.confidence if hasattr(det, 'confidence') else 1.0
            feat = detection_features[i] if detection_features is not None else None

            if conf > CONF_THRESH:
                high_score_dets.append(det)
                high_score_feats.append(feat)
                high_score_indices.append(i)
            elif conf > 0.1:  # 忽略极低分检测
                low_score_dets.append(det)
                low_score_feats.append(feat)

        # 2. Stage 1: 匹配高分检测 (Cascade匹配: 距离+方向+外观)
        # 使用所有现有轨迹进行匹配
        all_track_indices = list(range(len(self.tracks)))

        matches_h, u_track_h, u_det_h = self._match_cascade(
            all_track_indices, high_score_dets, high_score_feats
        )

        # 3. Stage 2: 匹配低分检测 (仅距离匹配)
        # 仅尝试匹配 Stage 1 中未匹配的、且状态为 CONFIRMED 的轨迹
        # (未确认的轨迹如果不匹配高分检测，通常就是误检，不应该去匹配低分检测)
        r_track_indices = [
            t_idx for t_idx in u_track_h
            if self.tracks[t_idx].state == KalmanTrackState.Confirmed
        ]

        matches_l, u_track_l, u_det_l = self._match_distance_only(
            r_track_indices, low_score_dets
        )

        # 4. 更新轨迹状态

        # 4.1 更新 Stage 1 匹配成功的 (高分)
        for t_idx, d_idx in matches_h:
            det = high_score_dets[d_idx]
            feat = high_score_feats[d_idx]
            x = det.center[0] if hasattr(det, 'center') else det[0]
            y = det.center[1] if hasattr(det, 'center') else det[1]
            self.tracks[t_idx].update(x, y, feature=feat)

        # 4.2 更新 Stage 2 匹配成功的 (低分)
        for t_idx, d_idx in matches_l:
            det = low_score_dets[d_idx]
            # 低分检测通常不更新外观特征，只更新位置
            # 因为低分目标的特征往往受遮挡或模糊影响，会污染轨迹特征库
            x = det.center[0] if hasattr(det, 'center') else det[0]
            y = det.center[1] if hasattr(det, 'center') else det[1]
            self.tracks[t_idx].update(x, y, feature=None)  # 注意: feature=None

        # 4.3 处理未匹配的高分检测 -> 新建轨迹
        for d_idx in u_det_h:
            det = high_score_dets[d_idx]
            feat = high_score_feats[d_idx]
            x = det.center[0] if hasattr(det, 'center') else det[0]
            y = det.center[1] if hasattr(det, 'center') else det[1]

            new_track = KalmanTrack(x, y, n_init=self.n_init, max_age=self.max_age, feature=feat)
            self.tracks.append(new_track)

        # 4.4 处理未匹配的轨迹 (两阶段都没匹配上)
        # u_track_h 中包含了 'Tentative' 的未匹配轨迹
        # u_track_l 中包含了 'Confirmed' 的未匹配轨迹
        # 还要加上那些 Tentative 且没在 Stage 1 匹配上的

        # 逻辑梳理:
        # Stage 2 没匹配上的 Confirmed 轨迹 -> u_track_l
        # Stage 1 没匹配上的 Tentative 轨迹 -> 需要从 u_track_h 中找出 Tentative 的

        unmatched_confirmed = set(u_track_l)
        unmatched_tentative = set([t for t in u_track_h if self.tracks[t].state == KalmanTrackState.Tentative])

        final_unmatched_tracks = unmatched_confirmed.union(unmatched_tentative)

        for t_idx in final_unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # 4.5 删除低分未匹配检测 (直接忽略)
        # pass

        # 5. 清理已删除轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 6. MOTA 评估 (仅用于调试)
        if gt_positions is not None and gt_ids is not None:
            # 为了公平评估，这里传入所有的 detections (或者只传入高分的?)
            # 通常 MOTA 计算只看高分检测，否则 FP 会很高
            eval_dets_pos = [(d.center[0], d.center[1]) for d in high_score_dets]
            self._update_mota_stats(eval_dets_pos, gt_positions, gt_ids)

        return [(t.track_id, t.position[0], t.position[1]) for t in self.tracks if t.is_confirmed()]

    def _update_mota_stats(self, detections, gt_positions, gt_ids):
        # ... (保持原有的统计逻辑不变) ...
        # 注意: 这里的 detections 是 (x,y) 元组列表
        num_gt = len(gt_positions)
        self.total_gt += num_gt
        if num_gt == 0:
            self.total_fp += len(detections)
            return
        if len(detections) == 0:
            self.total_fn += num_gt
            return

        cost_matrix = np.zeros((num_gt, len(detections)), dtype=np.float32)
        for g, (gt_x, gt_y) in enumerate(gt_positions):
            for d, (det_x, det_y) in enumerate(detections):
                cost_matrix[g, d] = np.sqrt((gt_x - det_x) ** 2 + (gt_y - det_y) ** 2)

        row_indices, col_indices = hungarian_algorithm(cost_matrix)

        matched_gt = set()
        matched_det = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.MOTA_EVAL_THRESHOLD:
                matched_gt.add(row)
                matched_det.add(col)
                self.total_matches += 1
                self.match_distances.append(cost_matrix[row, col])

                gt_id = gt_ids[row]
                if gt_id in self.gt_to_track_id:
                    if self.gt_to_track_id[gt_id] != col:
                        self.total_idsw += 1
                        self.gt_to_track_id[gt_id] = col
                else:
                    self.gt_to_track_id[gt_id] = col

        self.total_fn += num_gt - len(matched_gt)
        self.total_fp += len(detections) - len(matched_det)

        DEBUG_MOTA = os.environ.get('DEBUG_MOTA', '0') == '1'
        if DEBUG_MOTA:
            print(f"🔍 [MOTA] GT={num_gt}, Det={len(detections)}, Matched={len(matched_gt)}")

    def get_mota(self):
        if self.total_gt == 0: return 0.0
        return max(0.0, 1.0 - (self.total_fn + self.total_fp + self.total_idsw) / self.total_gt)

    def get_motp(self):
        if len(self.match_distances) == 0: return 0.0
        return max(0.0, 1.0 - np.mean(self.match_distances) / self.max_distance)


class SequenceMOTATracker:
    """序列级别的跟踪器管理器 - 确保不同视频序列之间不会混淆"""

    def __init__(self, max_age=5, n_init=2, max_distance=5.0):
        self.max_age = max_age
        self.n_init = n_init
        self.max_distance = max_distance
        self.trackers = {}
        self.current_group = None
        self.global_gt = 0
        self.global_fp = 0
        self.global_fn = 0
        self.global_idsw = 0
        self.global_matches = 0
        self.global_distances = []

    def get_tracker(self, group_name):
        if group_name not in self.trackers:
            self.trackers[group_name] = KalmanMOTATracker(
                max_age=self.max_age, n_init=self.n_init, max_distance=self.max_distance
            )
        return self.trackers[group_name]

    def update(self, group_name, frame_idx, detections, gt_positions=None, gt_ids=None, detection_features=None):
        """🔴 V45: 支持外观特征"""
        if self.current_group is not None and self.current_group != group_name:
            if self.current_group in self.trackers:
                self._accumulate_stats(self.trackers[self.current_group])

        self.current_group = group_name
        tracker = self.get_tracker(group_name)

        if frame_idx == 0 or (hasattr(tracker, 'last_frame') and frame_idx < tracker.last_frame):
            tracker.reset()

        tracker.last_frame = frame_idx
        return tracker.update(detections, gt_positions, gt_ids, detection_features=detection_features)

    def _accumulate_stats(self, tracker):
        self.global_gt += tracker.total_gt
        self.global_fp += tracker.total_fp
        self.global_fn += tracker.total_fn
        self.global_idsw += tracker.total_idsw
        self.global_matches += tracker.total_matches
        self.global_distances.extend(tracker.match_distances)

    def get_global_mota(self):
        if self.current_group in self.trackers:
            tracker = self.trackers[self.current_group]
            total_gt = self.global_gt + tracker.total_gt
            total_fp = self.global_fp + tracker.total_fp
            total_fn = self.global_fn + tracker.total_fn
            total_idsw = self.global_idsw + tracker.total_idsw
        else:
            total_gt = self.global_gt
            total_fp = self.global_fp
            total_fn = self.global_fn
            total_idsw = self.global_idsw

        if total_gt == 0:
            return 0.0
        mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt
        return max(0.0, mota)

    def get_global_motp(self):
        if self.current_group in self.trackers:
            all_distances = self.global_distances + self.trackers[self.current_group].match_distances
        else:
            all_distances = self.global_distances
        if len(all_distances) == 0:
            return 0.0
        avg_distance = np.mean(all_distances)
        return max(0.0, 1.0 - (avg_distance / self.max_distance))

    def reset_all(self):
        self.trackers = {}
        self.current_group = None
        self.global_gt = 0
        self.global_fp = 0
        self.global_fn = 0
        self.global_idsw = 0
        self.global_matches = 0
        self.global_distances = []
        KalmanTrack._next_id = 1

    def print_global_metrics(self):
        mota = self.get_global_mota()
        motp = self.get_global_motp()
        if self.current_group in self.trackers:
            tracker = self.trackers[self.current_group]
            total_gt = self.global_gt + tracker.total_gt
            total_fp = self.global_fp + tracker.total_fp
            total_fn = self.global_fn + tracker.total_fn
            total_idsw = self.global_idsw + tracker.total_idsw
            total_matches = self.global_matches + tracker.total_matches
        else:
            total_gt = self.global_gt
            total_fp = self.global_fp
            total_fn = self.global_fn
            total_idsw = self.global_idsw
            total_matches = self.global_matches

        print(f"\n{'=' * 60}")
        print(f"📊 全局MOTA统计 [卡尔曼跟踪器, 匹配阈值={self.max_distance}m]:")
        print(f"   序列数: {len(self.trackers)}")
        print(f"   总GT={total_gt}, 总匹配={total_matches}")
        print(f"   总FP={total_fp}, 总FN={total_fn}, 总IDSW={total_idsw}")
        print(f"   ✅ MOTA={mota * 100:.2f}%, MOTP={motp * 100:.2f}%")
        print(f"{'=' * 60}\n")


print("✅ 卡尔曼滤波跟踪器已加载（封装到fusion_model.py）")