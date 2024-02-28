import os, sys
import numpy as np
from mcpd.articulated_registration import ArtRegistration
from scipy.optimize import linear_sum_assignment
from functools import partial
from argparse import ArgumentParser
import trimesh
    
    
def test_metrics(labels, Z):
    num = np.unique(labels, axis=0)
    num = num.shape[0]
    
    K = Z.shape[1]
    select_K = np.asarray(np.argmax(Z, axis=1))
    
    iou_mtx = np.zeros((num, K))
    for i in range(num):
        for j in range(K):
            tp = np.sum((select_K == j) * (labels == i))
            fp = np.sum((select_K == j) * (labels != i))
            fn = np.sum((select_K != j) * (labels == i))

            # % if current sugment exists then count the iou
            iou = (tp+1e-12) / (tp+fp+fn+1e-12)
            iou_mtx[i, j] = iou

    row_ind, col_ind = linear_sum_assignment(-iou_mtx)
    
    return np.mean(iou_mtx[row_ind, col_ind])


def visualize_multiple_images(iteration, error, X, Y, TY, Z, P=None, dist=None, save=False, data_dir=None):
    global save_image_idx
    select_K = np.argmax(Z, axis=1)
    nPoints = Y.shape[0]
    pts_mtx = np.copy(Y)
    color_mtx = np.zeros((nPoints, 3))
        
    color = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],[0,0,128]]

    for i in range(nPoints):
        k = select_K[i]
        color_mtx[i, :] = np.array(color[k]) / 255.0

    pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
    scene = trimesh.Scene()
    scene.add_geometry(pcl)
    scene.show(line_settings={'point_size':10})
            
    for t in range(TY.shape[0]):
        for i in range(nPoints):
            k = select_K[i]
            color_mtx[i, :] = np.array(color[k]) / 255.0
            pts_mtx[i, :] = TY[t, k, i, :]

        scene = trimesh.Scene()
        pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
        scene.add_geometry(pcl)
        scene.show(line_settings={'point_size':10})


def visualize_multiple_images(iteration, error, X, Y, TY, Z, P=None, dist=None, save=False, data_dir=None):
    global save_image_idx
    select_K = np.argmax(Z, axis=1)
    nPoints = Y.shape[0]
    pts_mtx = np.copy(Y)
    color_mtx = np.zeros((nPoints, 3))
        
    color = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],[0,0,128]]

    for i in range(nPoints):
        k = select_K[i]
        color_mtx[i, :] = np.array(color[k]) / 255.0

    pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
    scene = trimesh.Scene()
    scene.add_geometry(pcl)
    scene.show(line_settings={'point_size':10})
            
    for t in range(TY.shape[0]):
        for i in range(nPoints):
            k = select_K[i]
            color_mtx[i, :] = np.array(color[k]) / 255.0
            pts_mtx[i, :] = TY[t, k, i, :]

        scene = trimesh.Scene()
        pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
        scene.add_geometry(pcl)
        scene.show(line_settings={'point_size':10})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_mixture', type=int, default=5)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--vis_interval", type=int, default=500)
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--datadir", type=str, default="data/glasses0")
    parser.add_argument("--savedir", type=str, default="results/glasses0")
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--torch", action='store_true')

    args = parser.parse_args(sys.argv[1:])

    num_mixture = args.num_mixture
    num_iterations = args.num_iterations
    vis = args.vis
    vis_interval = args.vis_interval
    datadir = args.datadir
    savedir = args.savedir
    save = args.save
    torch = args.torch

    data_list = os.listdir(datadir)
    source_point_path, target_point_path = os.path.join(datadir, data_list[0]), os.path.join(datadir, data_list[1])
    source_point_npz = np.load(source_point_path)
    target_point_npz = np.load(target_point_path)

    ys = source_point_npz["sampled_points"]
    xs = target_point_npz["sampled_points"]

    reg = ArtRegistration(ys, xs, num_mixture, max_iterations=num_iterations, vis_interval=vis_interval, vis=vis, gpu=torch)
    callback = partial(visualize_multiple_images)
    TY, params = reg.register(callback)
    
    if save:
        R, t, Z = params
        os.makedirs(savedir, exist_ok=True)
        save_path = os.path.join(savedir, "results.npz")
        np.savez(save_path, X=xs, Y=ys, TY=TY, R=R, t=t, Z=Z)