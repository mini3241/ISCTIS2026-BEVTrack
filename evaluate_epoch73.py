#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理评估脚本 - epoch_73 在 valid.txt (2244样本) 上评估
以 train_v42 为主，包含跟踪头
"""

import os
import sys

os.chdir('/mnt/ChillDisk/personal_data/mij/pythonProject1')

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 先导入fusion_model，覆盖BEV配置
import fusion_model_v42_with_yolo as fusion_module
fusion_module.BEV_X_MIN = -35.0
fusion_module.BEV_X_MAX = 35.0
fusion_module.BEV_Y_MIN = 0.0
fusion_module.BEV_Y_MAX = 70.0

# 从train脚本导入（以train为主）
from train_v42_joint_fixed_old_version_v3_resume_continue import (
    MultiModalLabeledDataset,
    custom_collate_fn,
    compute_mota_motp,
    BEV_X_MIN, BEV_X_MAX, BEV_Y_MIN, BEV_Y_MAX,
    SequenceMOTATracker,  # 跟踪器
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 配置
CHECKPOINT_PATH = './checkpoints_label_v2/model_epoch_73.pth'
MAPPING_CSV = '/mnt/ourDataset_v2/mapping.csv'
VALID_TXT = '/mnt/ourDataset_v2/valid.txt'
DATA_ROOT = '/mnt/ourDataset_v2/ourDataset_v2_label'
IMG_SIZE = 256
BATCH_SIZE = 1
NUM_WORKERS = 4

print(f"\n{'='*70}")
print(f"推理评估 - Epoch 73 (含跟踪头)")
print(f"{'='*70}")
print(f"BEV范围: X=[{BEV_X_MIN}, {BEV_X_MAX}], Y=[{BEV_Y_MIN}, {BEV_Y_MAX}]")
print(f"{'='*70}\n")


def extract_detections_from_output(output, H, W):
    """从单个样本的模型输出提取检测结果"""
    if len(output.shape) == 3:
        heatmap = output[0].cpu().numpy()
    else:
        heatmap = output.cpu().numpy()

    # Sigmoid
    heatmap = np.clip(heatmap, -50, 50)
    heatmap = 1.0 / (1.0 + np.exp(-heatmap))

    # MaxPool NMS
    heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    max_pooled = F.max_pool2d(heatmap_t, kernel_size=3, stride=1, padding=1)
    peak_mask = (heatmap_t == max_pooled).float()
    filtered = (heatmap_t * peak_mask).squeeze().numpy()

    # Top-K
    max_k = 15
    threshold = 0.2

    flat = filtered.flatten()
    k = min(max_k, len(flat))
    top_k_idx = np.argpartition(flat, -k)[-k:]
    top_k_val = flat[top_k_idx]

    valid_idx = top_k_idx[top_k_val >= threshold]
    coords = np.unravel_index(valid_idx, heatmap.shape)

    # 坐标转换
    detections = []
    hH, hW = heatmap.shape
    for y, x in zip(coords[0], coords[1]):
        norm_y = y / (hH - 1) if hH > 1 else 0.5
        norm_x = x / (hW - 1) if hW > 1 else 0.5
        world_x = norm_x * (BEV_X_MAX - BEV_X_MIN) + BEV_X_MIN
        world_y = norm_y * (BEV_Y_MAX - BEV_Y_MIN) + BEV_Y_MIN
        conf = float(heatmap[y, x])
        detections.append((world_x, world_y, conf))

    # 距离NMS
    final_dets = []
    for det in detections:
        too_close = False
        for ex in final_dets:
            dist = np.sqrt((det[0]-ex[0])**2 + (det[1]-ex[1])**2)
            if dist < 5.0:
                if det[2] <= ex[2]:
                    too_close = True
                else:
                    final_dets.remove(ex)
                break
        if not too_close:
            final_dets.append(det)

    return [(d[0], d[1]) for d in final_dets]  # 返回 [(x, y), ...]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print("加载模型...")
    model = fusion_module.FusionNet().to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"加载成功: epoch={ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 数据集
    print("加载数据集...")
    dataset = MultiModalLabeledDataset(
        mapping_csv=MAPPING_CSV,
        id_txt=VALID_TXT,
        data_root=DATA_ROOT,
        img_size=IMG_SIZE,
        use_camera='LeopardCamera0'
    )

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn
    )

    # 初始化跟踪器 - 最优配置
    print("初始化跟踪器...")
    tracker = SequenceMOTATracker(max_age=1, n_init=1, max_distance=5.0)

    # 用于收集轨迹的列表
    all_pred_tracks = []  # 跟踪后的轨迹
    all_gt_tracks = []
    all_raw_detections = []  # 原始检测（用于计算Detection MOTA）

    # 记录每个group的帧索引
    group_frame_counters = {}

    print("开始推理...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="推理")):
            if batch is None:
                continue

            inputs = tuple(t.to(device) for t in batch['model_inputs'])
            targets_batch = batch['targets']
            frame_ids = batch['frame_ids']
            group_names = batch.get('group_names', ['default'] * len(frame_ids))

            out = model(*inputs)
            outputs = out[0] if isinstance(out, tuple) else out

            if batch_idx == 0:
                print(f"\n第1批: outputs.shape={outputs.shape}")

            # 处理每个样本
            for b in range(outputs.shape[0]):
                output = outputs[b]
                targets = targets_batch[b] if b < len(targets_batch) else []
                frame_id = str(frame_ids[b])
                group_name = group_names[b] if b < len(group_names) else 'default'

                # 获取当前group的帧索引
                if group_name not in group_frame_counters:
                    group_frame_counters[group_name] = 0
                frame_idx = group_frame_counters[group_name]
                group_frame_counters[group_name] += 1

                # 提取检测
                detections = extract_detections_from_output(output, IMG_SIZE, IMG_SIZE)

                # 收集原始检测（用于Detection MOTA）
                for dx, dy in detections:
                    all_raw_detections.append((frame_id, dx, dy))

                # 提取GT
                gt_positions = []
                gt_ids = []
                for t in targets:
                    if isinstance(t, dict) and 'object_id' in t:
                        x, y = float(t.get('x', 0)), float(t.get('y', 0))
                        if BEV_X_MIN <= x <= BEV_X_MAX and BEV_Y_MIN <= y <= BEV_Y_MAX:
                            gt_positions.append((x, y))
                            gt_ids.append(t['object_id'])

                # 更新跟踪器
                confirmed_tracks = tracker.update(
                    group_name=group_name,
                    frame_idx=frame_idx,
                    detections=detections,
                    gt_positions=gt_positions if gt_positions else None,
                    gt_ids=gt_ids if gt_ids else None
                )

                # 收集预测轨迹 - confirmed_tracks返回的是(track_id, x, y)元组
                for track_tuple in confirmed_tracks:
                    track_id, x, y = track_tuple
                    all_pred_tracks.append((frame_id, track_id, x, y, 2.0, 2.0, 1.0, group_name))

                # 收集GT轨迹
                for gt_id, (gx, gy) in zip(gt_ids, gt_positions):
                    all_gt_tracks.append((frame_id, gt_id, gx, gy, 2.0, 2.0, group_name))

    # 最终累积最后一个group的统计
    if tracker.current_group in tracker.trackers:
        tracker._accumulate_stats(tracker.trackers[tracker.current_group])

    # 输出结果
    print(f"\n{'='*70}")
    print(f"跟踪器统计结果")
    print(f"{'='*70}")
    print(f"总GT数: {tracker.global_gt}")
    print(f"总匹配数: {tracker.global_matches}")
    print(f"总FP数: {tracker.global_fp}")
    print(f"总FN数: {tracker.global_fn}")
    print(f"总IDSW数: {tracker.global_idsw}")

    if tracker.global_gt > 0:
        mota = 1.0 - (tracker.global_fp + tracker.global_fn + tracker.global_idsw) / tracker.global_gt
        mota = max(0.0, mota)
        motp = np.mean(tracker.global_distances) if tracker.global_distances else 0.0

        print(f"\n{'='*70}")
        print(f"最终结果 (含跟踪头)")
        print(f"{'='*70}")
        print(f"MOTA: {mota*100:.2f}%")
        print(f"MOTP: {motp:.2f}m")
        print(f"{'='*70}")
    else:
        print("无GT数据")

    # ========== 保存轨迹数据用于绘图 ==========
    save_dir = './track_results_epoch73'
    os.makedirs(save_dir, exist_ok=True)

    # 建立 group_name -> group_idx 的映射
    unique_groups = sorted(set(p[7] for p in all_pred_tracks) | set(g[6] for g in all_gt_tracks))
    group_name_to_idx = {name: idx for idx, name in enumerate(unique_groups)}

    # 由于frame_id是原始字符串ID，需要转为组内帧序号
    # 重新按group分组计算frame_idx
    group_frame_map = {}  # {group_name: {frame_id: frame_idx}}
    for p in all_pred_tracks:
        gn = p[7]
        fid = str(p[0])
        if gn not in group_frame_map:
            group_frame_map[gn] = {}
        if fid not in group_frame_map[gn]:
            group_frame_map[gn][fid] = len(group_frame_map[gn])
    for g in all_gt_tracks:
        gn = g[6]
        fid = str(g[0])
        if gn not in group_frame_map:
            group_frame_map[gn] = {}
        if fid not in group_frame_map[gn]:
            group_frame_map[gn][fid] = len(group_frame_map[gn])

    est_list = []
    for p in all_pred_tracks:
        frame_id, track_id, x, y = p[0], p[1], p[2], p[3]
        group_name = p[7]
        group_idx = group_name_to_idx[group_name]
        frame_idx = group_frame_map[group_name][str(frame_id)]
        est_list.append([float(track_id), float(x), float(y), 0.0, float(frame_idx), float(group_idx)])

    all_estimated_tracks_np = np.array(est_list, dtype=np.float64) if est_list else np.zeros((0, 6))

    # all_gt_tracks.npy: (gt_id, x, y, z, class, frame_idx, group_idx)
    gt_list = []
    for g in all_gt_tracks:
        # g = (frame_id, gt_id, x, y, w, h, group_name)
        frame_id, gt_id, x, y = g[0], g[1], g[2], g[3]
        group_name = g[6]
        group_idx = group_name_to_idx[group_name]
        frame_idx = group_frame_map[group_name][str(frame_id)]
        gt_list.append([float(gt_id), float(x), float(y), 0.0, 1.0, float(frame_idx), float(group_idx)])

    all_gt_tracks_np = np.array(gt_list, dtype=np.float64) if gt_list else np.zeros((0, 7))

    # group_names.npy: 保存 (group_idx, group_name) 的列表
    group_names_list = np.array([[idx, name] for name, idx in sorted(group_name_to_idx.items(), key=lambda x: x[1])],
                                dtype=object)

    np.save(os.path.join(save_dir, 'all_estimated_tracks.npy'), all_estimated_tracks_np)
    np.save(os.path.join(save_dir, 'all_gt_tracks.npy'), all_gt_tracks_np)
    np.save(os.path.join(save_dir, 'group_names.npy'), group_names_list)

    print(f"\n轨迹数据已保存到 {save_dir}/")
    print(f"  all_estimated_tracks.npy: shape={all_estimated_tracks_np.shape}")
    print(f"  all_gt_tracks.npy: shape={all_gt_tracks_np.shape}")
    print(f"  group_names.npy: {len(group_names_list)} groups")
    # ========== 保存轨迹数据完毕 ==========

    # 同时计算Detection MOTA作为对比（使用原始检测，不是跟踪输出）
    print(f"\n--- Detection MOTA (原始检测，不含跟踪) ---")
    if all_raw_detections and all_gt_tracks:
        from scipy.optimize import linear_sum_assignment

        pred_by_frame = {}
        gt_by_frame = {}
        for p in all_raw_detections:
            fid = str(p[0])
            if fid not in pred_by_frame:
                pred_by_frame[fid] = []
            pred_by_frame[fid].append((p[1], p[2]))  # x, y
        for g in all_gt_tracks:
            fid = str(g[0])
            if fid not in gt_by_frame:
                gt_by_frame[fid] = []
            gt_by_frame[fid].append((g[2], g[3]))

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_dist = 0

        for fid in set(list(pred_by_frame.keys()) + list(gt_by_frame.keys())):
            preds = pred_by_frame.get(fid, [])
            gts = gt_by_frame.get(fid, [])
            total_gt += len(gts)

            if len(preds) == 0:
                total_fn += len(gts)
                continue
            if len(gts) == 0:
                total_fp += len(preds)
                continue

            cost = np.zeros((len(gts), len(preds)))
            for i, g in enumerate(gts):
                for j, p in enumerate(preds):
                    cost[i, j] = np.sqrt((g[0]-p[0])**2 + (g[1]-p[1])**2)

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_gt = set()
            matched_pred = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= 5.0:
                    total_tp += 1
                    total_dist += cost[r, c]
                    matched_gt.add(r)
                    matched_pred.add(c)

            total_fn += len(gts) - len(matched_gt)
            total_fp += len(preds) - len(matched_pred)

        det_mota = 1.0 - (total_fp + total_fn) / total_gt if total_gt > 0 else 0
        det_motp = total_dist / total_tp if total_tp > 0 else 0

        print(f"GT总数: {total_gt}, TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"Detection MOTA: {det_mota*100:.2f}%")
        print(f"Detection MOTP: {det_motp:.2f}m")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
