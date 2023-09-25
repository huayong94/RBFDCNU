import torch
from scipy.ndimage import morphology
import numpy as np


class DistDistance(torch.nn.Module):
    def __init__(self):
        super(DistDistance, self).__init__()
        self.pdist = torch.nn.PairwiseDistance(2)

    def forward(self, contour1, contour2, resolution):
        len1 = contour1.size(1)
        len2 = contour2.size(1)
        dim = contour1.size(2)

        contour1_t = contour1.unsqueeze(1).repeat(1, len2, 1, 1).view(-1, dim)
        contour2_t = contour2.unsqueeze(2).repeat(1, 1, len1, 1).view(-1, dim)

        contour2_to_contour1_each_dist = self.pdist(contour1_t,
                                                    contour2_t).view(
                                                        -1, len2, len1)

        contour2_min_dist, _ = torch.min(contour2_to_contour1_each_dist, dim=2)

        return torch.mean(contour2_min_dist * resolution, dim=1)


class HausdorffDistance(torch.nn.Module):
    def __init__(self):
        super(HausdorffDistance, self).__init__()
        self.pdist = torch.nn.PairwiseDistance(2)

    def supinf(self, contour1, contour2):
        len1 = contour1.size(1)
        len2 = contour2.size(1)
        dim = contour1.size(2)

        contour1_t = contour1.unsqueeze(1).repeat(1, len2, 1, 1).view(-1, dim)
        contour2_t = contour2.unsqueeze(2).repeat(1, 1, len1, 1).view(-1, dim)

        contour2_to_contour1_each_dist = self.pdist(contour1_t,
                                                    contour2_t).view(
                                                        -1, len2, len1)

        contour2_min_dist, _ = torch.min(contour2_to_contour1_each_dist, dim=2)

        return torch.max(contour2_min_dist, dim=1)[0].unsqueeze(1)

    def forward(self, contour1, contour2, resolution):
        h21 = self.supinf(contour1, contour2)
        h12 = self.supinf(contour2, contour1)

        return torch.max(torch.cat([h12, h21], 1), dim=1)[0] * resolution


class SurfaceDistanceFromSeg(object):
    def __init__(self, connectivity=1, ndim=2):
        self.conn = morphology.generate_binary_structure(ndim, connectivity)
        self.show = 1

    def compute_surface_distances(self,
                                  seg_gt: np.ndarray,
                                  seg_pred: np.ndarray,
                                  spacing_mm=1):
        seg_gt = seg_gt.astype(np.bool)
        seg_pred = seg_pred.astype(np.bool)

        borders_gt = seg_gt ^ morphology.binary_erosion(seg_gt, self.conn)
        borders_pred = seg_pred ^ morphology.binary_erosion(
            seg_pred, self.conn)

        if borders_gt.any():
            distmap_gt = morphology.distance_transform_edt(
                ~borders_gt, spacing_mm)
        else:
            distmap_gt = np.Inf * np.ones_like(borders_gt)

        if borders_pred.any():
            distmap_pred = morphology.distance_transform_edt(
                ~borders_pred, spacing_mm)
        else:
            distmap_pred = np.Inf * np.ones_like(borders_pred)

        dist_gt_to_pred = distmap_pred[borders_gt]
        dist_pred_to_gt = distmap_gt[borders_pred]

        return {
            "dist_gt_to_pred": sorted(dist_gt_to_pred),
            "dist_pred_to_gt": sorted(dist_pred_to_gt)
        }

    def compute_average_surface_distance(self, surface_distances):
        average_distance_gt_to_pred = np.mean(
            surface_distances['dist_gt_to_pred'])
        average_distance_pred_to_gt = np.mean(
            surface_distances['dist_pred_to_gt'])
        return average_distance_gt_to_pred, average_distance_pred_to_gt

    def compute_robust_hausdorff(self, surface_distances, percent=95):
        distances_gt_to_pred = surface_distances["dist_gt_to_pred"]
        distances_pred_to_gt = surface_distances["dist_pred_to_gt"]

        if len(distances_gt_to_pred) > 0:
            idx = len(distances_gt_to_pred) * percent // 100 - 1
            perc_distances_gt_to_pred = distances_gt_to_pred[idx]
            # if self.show:
            #     print(distances_gt_to_pred)
            #     print(idx, len(distances_gt_to_pred) - 1)
            #     self.show = 0
        else:
            perc_distances_gt_to_pred = np.Inf

        if len(distances_pred_to_gt) > 0:
            idx = len(distances_pred_to_gt) * percent // 100 - 1
            perc_distances_pred_to_gt = distances_pred_to_gt[idx]
        else:
            perc_distances_pred_to_gt = np.Inf

        return max(perc_distances_gt_to_pred, perc_distances_pred_to_gt)


class MaxMinPointDist(torch.nn.Module):
    def __init__(self, point_num,max_v=10e4):
        super(MaxMinPointDist, self).__init__()
        bigeye = torch.eye(point_num, point_num) * max_v
        self.register_buffer('bigeye', bigeye.float())
        self.point_num = point_num

    def forward(self, point):

        point1 = point.unsqueeze(1).repeat(1, self.point_num, 1, 1).float()
        point2 = point.unsqueeze(2).repeat(1, 1, self.point_num, 1).float()

        dist = torch.norm(point1 - point2, dim=3) + self.bigeye

        min_dist, _ = torch.min(dist, dim=2)

        c = torch.max(min_dist, dim=1)[0]
        mask = torch.tensor([8]).cuda()
        c = torch.max(c, mask)
 
        return c