# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
# Run with python -m scripts.test_3dmatch_refactor
import os
import sys
import math
import logging
import open3d as o3d
import numpy as np
import time
import torch
import copy
import tqdm
import pysnooper
from loguru import logger 
import PcdVis

sys.path.append('.')
import MinkowskiEngine as ME
from config import get_config
from model import load_model

from dataloader.data_loaders import ThreeDMatchTrajectoryDataset
from core.knn import find_knn_gpu
from core.deep_global_registration import DeepGlobalRegistration

from util.timer import Timer
from util.pointcloud import make_open3d_point_cloud,get_matching_indices

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])

# Criteria
def rte_rre(T_pred, T_gt, rte_thresh, rre_thresh, eps=1e-16):
  if T_pred is None:
    return np.array([0, np.inf, np.inf])

  rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
  rre = np.arccos(
      np.clip((np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1 + eps,
              1 - eps)) * 180 / math.pi
  return np.array([rte < rte_thresh and rre < rre_thresh, rte, rre])

def inlier_ratio_core(points_src, points_tgt, feats_src, feats_tgt, transf, inlier_threshold=0.1):
    R, t = transf[:3, :3], transf[:3, 3]
    dists = np.matmul(feats_src, feats_tgt.T) # (n, m)
    row_max_inds = np.argmax(dists, axis=1)
    col_max_inds = np.argmax(dists, axis=0)
    points_src = points_src @ R.T + t
    inlier_mask = np.sum((points_src - points_tgt[row_max_inds]) ** 2, axis=1) < inlier_threshold ** 2
    inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)
    
    # mutual inlier ratio
    mutual_corrs = []
    for i in range(len(points_src)):
        if col_max_inds[row_max_inds[i]] == i:
            mutual_corrs.append([i, row_max_inds[i]])
    mutual_corrs = np.array(mutual_corrs, dtype=np.int32)
    mutual_mask = np.sum((points_src[mutual_corrs[:, 0]] - points_tgt[mutual_corrs[:, 1]]) ** 2, axis=1) < inlier_threshold ** 2
    mutual_inlier_ratio = np.sum(mutual_mask) / len(mutual_corrs)
    return np.array([inlier_ratio, inlier_ratio > 0.05]) 
  
def registration_recall_core(points_src, points_tgt, coors, pred_T):
    points_src = points_src[coors[:, 0]]
    points_tgt = points_tgt[coors[:, 1]]
    R, t = pred_T[:3, :3], pred_T[:3, 3]
    points_src = points_src @ R.T + t
    mse = np.mean(np.sum((points_src - points_tgt) ** 2, axis=1))
    rmse = np.sqrt(mse)
    
    return np.array([rmse < 0.2, rmse])


def analyze_stats(stats, mask, method_names):
  mask = (mask > 0).squeeze(1)
  stats = stats[:, mask, :]

  print('Total result mean')
  for i, method_name in enumerate(method_names):
    print(method_name)
    print(stats[i].mean(0))

  print('Total successful result mean')
  for i, method_name in enumerate(method_names):
    sel = stats[i][:, 0] > 0
    sel_stats = stats[i][sel]
    print(method_name)
    print(sel_stats.mean(0))

def analyze_stats_tqdm(stats, mask, method_names, pbar):
  mask = (mask > 0).squeeze(1)
  stats = stats[:, mask, :]

  pbar.write('Total result mean')
  for i, method_name in enumerate(method_names):
    pbar.write(method_name)
    pbar.write(f"{stats[i].mean(0)}")

  pbar.write('Total successful result mean')
  for i, method_name in enumerate(method_names):
    sel = stats[i][:, 0] > 0
    sel_stats = stats[i][sel]
    pbar.write(method_name)
    pbar.write(f"{sel_stats.mean(0)}")


def create_pcd(xyz, color):
  # n x 3
  n = xyz.shape[0]
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  pcd.paint_uniform_color (color)
  pcd.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  return pcd


def draw_geometries_flip(pcds):
  pcds_transform = []
  flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
  for pcd in pcds:
    pcd_temp = copy.deepcopy(pcd)
    pcd_temp.transform(flip_transform)
    pcds_transform.append(pcd_temp)
  o3d.visualization.draw_geometries(pcds_transform)

@logger.catch 
def print_method_scenevals(subset_names, stats, i):
  import prettytable
  table = prettytable.PrettyTable(['scene_name', 'sr', 'rre', 'rte', 'time', 'ir', 'fmr', 'rr','rmse'])
  scene_vals = np.zeros((len(subset_names), 8))
  for sid, sname in enumerate(subset_names):
    curr_scene = stats[i, :, 8] == sid
    scene_vals[sid] = np.round((stats[i, curr_scene, :8]).mean(0),4)
    tablerow = [sname] + scene_vals[sid].tolist()
    table.add_row(tablerow)
  array = np.round(scene_vals.mean(0),4)
  tablerow = ['mean'] + array.tolist()
  table.add_row(tablerow)
  print(table)
  print('All scenes')
  print(scene_vals)
  print('Scene average')
  print(scene_vals.mean(0))
  
  
def evaluate(methods, method_names, data_loader, config, debug=False):
  tot_num_data = len(data_loader.dataset)
  data_loader_iter = iter(data_loader)

  # Accumulate success, rre, rte, time, sid
  mask = np.zeros((tot_num_data, 1)).astype(int)
  stats = np.zeros((len(methods), tot_num_data, 9))

  dataset = data_loader.dataset
  subset_names = open(dataset.DATA_FILES[dataset.phase]).read().split()

  pbar = tqdm.tqdm(range(tot_num_data),total=tot_num_data,desc='evaluating')
  for batch_idx in pbar:
    batch = data_loader_iter.next()

    # Skip too sparse point clouds
    sname, xyz0, xyz1, trans = batch[0]

    sid = subset_names.index(sname)
    T_gt = np.linalg.inv(trans)
    
    for i, method in enumerate(methods):
      start = time.time()
      T,res = method.register(xyz0, xyz1)
      end = time.time()
      feats0 = res['feats0']
      feats1 = res['feats1']
      
      point_src = res['points0']
      point_tgt = res['points1']
      
      coors = get_matching_indices(source=make_open3d_point_cloud(point_src), target=make_open3d_point_cloud(point_tgt), trans=T_gt, search_voxel_size= 0.0375)
      coors_filter = {}
      for coor1,coor2 in coors:
        if coor1 not in coors_filter:
          coors_filter[coor1] = coor2
      coors_filter = np.array([[coor1, coor2] for coor1, coor2 in coors_filter.items()])

      stats[i, batch_idx, :3] = rte_rre(T, T_gt, config.success_rte_thresh,
                                        config.success_rre_thresh)
      stats[i, batch_idx, 3] = end - start
      
      stats[i, batch_idx, 4:6] = inlier_ratio_core(point_src, point_tgt, feats0, feats1, T)
      stats[i, batch_idx, 6:8] = registration_recall_core(point_src, point_tgt, coors_filter, T)
      stats[i, batch_idx, 8] = sid
      
      mask[batch_idx] = 1
      if stats[i, batch_idx, 0] == 0:
        pbar.write(f"{method_names[i]}: failed")
      

    if batch_idx % 10 == 0 and batch_idx > 0:
      pbar.write('Summary {} / {}'.format(batch_idx, tot_num_data))
      analyze_stats_tqdm(stats, mask, method_names,pbar)

  # Save results
  filename = f'3dmatch-stats_{method.__class__.__name__}'
  if os.path.isdir(config.out_dir):
    out_file = os.path.join(config.out_dir, filename)
  else:
    out_file = filename  # save it on the current directory
  print(f'Saving the stats to {out_file}')
  np.savez(out_file, stats=stats, names=method_names)
  analyze_stats(stats, mask, method_names)

  # Analysis per scene
  for i, method in enumerate(methods):
    print(f'Scene-wise mean {method_names[i]}')
    print_method_scenevals(subset_names, stats, i)

if __name__ == '__main__':
  config = get_config()
  print(config)

  dgr = DeepGlobalRegistration(config)

  methods = [dgr]
  method_names = ['DGR']

  dset = ThreeDMatchTrajectoryDataset(phase='test',
                                      transform=None,
                                      random_scale=False,
                                      random_rotation=False,
                                      config=config)

  data_loader = torch.utils.data.DataLoader(dset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            collate_fn=lambda x: x,
                                            pin_memory=False,
                                            drop_last=True)

  evaluate(methods, method_names, data_loader, config, debug=False)
