import open3d as o3d
import numpy as np
from distinctipy import distinctipy

class PcdVis:
    def __init__(self, pcd_numbers):
        self.pcd_numbers = pcd_numbers
        self.colors = distinctipy.get_colors(pcd_numbers)
        self.pcds = []
    
    def add_pcd(self, xyz):
        self.pcds.append(create_pcd(xyz, self.colors[len(self.pcds)]))
    
    def add_pcds(self, args):
        for xyz in args:
            self.add_pcd(xyz)
        
    def add_pcd_w_corrs(self, xyz, corrs):
        self.pcds.append(create_pcd(xyz, self.colors[len(self.pcds)]))
        self.pcds.append(create_pcd(xyz[corrs], self.colors[len(self.pcds)]))
        
    def get_pcd_trans(self, xyz, T):
        R, t = T[:3, :3], T[:3, 3]
        point_trans = xyz @ R.T + t
        return point_trans
        
    def to_file(self, path, left=0, right=-1):
        if right == -1:
            right = len(self.pcds)
        res = o3d.geometry.PointCloud()
        for i in range(left, right):
            res = res + self.pcds[i]
        o3d.io.write_point_cloud(path, res)
        


def create_pcd(xyz, color):
  # n x 3
  n = xyz.shape[0]
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  pcd.paint_uniform_color (color)
  pcd.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  return pcd
  