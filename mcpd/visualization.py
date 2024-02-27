import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os
import open3d as o3d
import trimesh

color = [[0, 0, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [0, 255, 0], [255, 0, 255],
        [128, 255, 0], [255, 0, 128], [255, 128, 0], [0, 255, 128]]

save_image_idx = 0

def visualize_source_images(iteration, error, Y, initZ, Z, save=False, data_dir=None):
    global save_image_idx
    select_K = np.argmax(Z, axis=1)
    nPoints = Y.shape[0]
    pts_mtx = np.copy(Y)
    color_mtx = np.zeros((nPoints, 3))
    for i in range(nPoints):
        k = select_K[i]
        color_mtx[i, :] = np.array(color[k]) / 255.0
    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
    pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)
    
    if save:
        if data_dir is None:
            data_dir = "results/00010_frame/100_"
        print("save")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd_trans)
        vis.run()
        image = vis.capture_screen_float_buffer()
        plt.imsave(data_dir + "%05d.png"%(save_image_idx), np.asarray(image))
        vis.destroy_window()
        save_image_idx += 1
    else:
        print("visualize")
        o3d.visualization.draw_geometries([pcd_trans])
        
    select_K = np.argmax(initZ, axis=1)
    nPoints = Y.shape[0]
    pts_mtx = np.copy(Y)
    color_mtx = np.zeros((nPoints, 3))
    for i in range(nPoints):
        k = select_K[i]
        color_mtx[i, :] = np.array(color[k]) / 255.0
    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
    pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)
    
    if save:
        if data_dir is None:
            data_dir = "results/00010_frame/100_"
        print("save")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd_trans)
        vis.run()
        image = vis.capture_screen_float_buffer()
        plt.imsave(data_dir + "%05d.png"%(save_image_idx), np.asarray(image))
        vis.destroy_window()
        save_image_idx += 1
    else:
        print("visualize")
        o3d.visualization.draw_geometries([pcd_trans])
            
def visualize_image(iteration, error, X, Y, TY, Z, save=False):
    
    global save_image_idx
    select_K = np.argmax(Z, axis=1)
    nPoints = Y.shape[0]
    pts_mtx = np.copy(Y)
    color_mtx = np.zeros((nPoints, 3))
    for i in range(nPoints):
        k = select_K[i]
        color_mtx[i, :] = np.array(color[k]) / 255.0
    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
    pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    if save:
        print("save")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd_trans)
        vis.run()
        image = vis.capture_screen_float_buffer()
        #plt.imshow(np.asarray(image))
        #plt.show()
        plt.imsave("results/00010_step/500_%05d.png"%(save_image_idx), np.asarray(image))
        vis.destroy_window()
        save_image_idx += 1
    else:
        print("visualize")
        o3d.visualization.draw_geometries([pcd_trans])
        
    select_K = np.argmax(Z, axis=1)
    amax = np.amax(Z.numpy(), axis=1)
    
    nPoints = TY[0].shape[0]
    pts_mtx = np.zeros((nPoints, 3))
    color_mtx = np.zeros((nPoints, 3))
    for i in range(nPoints):
        k = select_K[i]
        color_mtx[i, :] = np.array(color[k]) / 255.0
        pts_mtx[i, :] = TY[k][i, :]
    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
    pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)
    
    if save:
        print("save")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd_trans)
        vis.run()
        image = vis.capture_screen_float_buffer()
        #plt.imshow(np.asarray(image))
        #plt.show()
        plt.imsave("results/00010_step/500_%05d.png"%(save_image_idx), np.asarray(image))
        vis.destroy_window()
        save_image_idx += 1
    else:
        print("visualize")
        o3d.visualization.draw_geometries([pcd, pcd_trans])
        
    """nPoints = Y[0].shape[0]
    pts_mtx = np.copy(ys)
    color_mtx = np.zeros((nPoints, 3))
    for i in range(nPoints):
        if amax[i] < 0.7:
            color_mtx[i, :] = [1.0, 0.0, 0.0]
        else:
            color_mtx[i, :] = [0.0, 0.0, 1.0]
    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
    pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)
    
    if save:
        print("save")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd_trans)
        vis.run()
        image = vis.capture_screen_float_buffer()
        #plt.imshow(np.asarray(image))
        #plt.show()
        plt.imsave("results/00010_step/500_%05d.png"%(save_image_idx), np.asarray(image))
        vis.destroy_window()
        save_image_idx += 1
    else:
        print("visualize")
        o3d.visualization.draw_geometries([pcd_trans])"""
        
    """pts_mtx = []
    color_mtx = []
    for i in range(nPoints):
        for k in range(K):
            if Z[i][k] > 0.3 and Z[i][k] < 0.5:
                #print(Z[i][k])
                color_mtx.append(np.array(color[k]) / 255.0)
                pts_mtx.append(Y[k][i, :].numpy())
    if len(pts_mtx) == 0:
        return
    color_mtx = np.asarray(color_mtx)
    pts_mtx = np.asarray(pts_mtx)
    pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
    pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)
    
    if save:
        print("save")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd_trans)
        vis.run()
        image = vis.capture_screen_float_buffer()
        #plt.imshow(np.asarray(image))
        #plt.show()
        plt.imsave("results/00010_step/500_%05d.png"%(save_image_idx), np.asarray(image))
        vis.destroy_window()
        save_image_idx += 1
    else:
        print("visualize")
        o3d.visualization.draw_geometries([pcd_trans])"""



def visualize_and_save_multiple_images(iteration, error, X, Y, TY, Z, P=None, dist=None, save=False, data_dir=None):
    global save_image_idx
    select_K = np.argmax(Z, axis=1)
    nPoints = Y.shape[0]
    pts_mtx = np.copy(Y)
    color_mtx = np.zeros((nPoints, 3))

    from scipy.spatial.transform import Rotation as R

    camera_rotation = np.eye(4)
    rotation = R.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix()
    rotation = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    camera_rotation[0:3, 0:3] = rotation
    camera = trimesh.scene.Camera(fov=(np.pi/4.0, np.pi/4.0))
    transform = camera.look_at([[0.0, 0.0, 0.0]], rotation=camera_rotation, distance=4.0)
    
    color = []
    for k in range(Z.shape[1]):
        color.append(255.0 * np.random.rand(3))
        
    color[0:14] = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],[0,0,128]]

    for i in range(nPoints):
        #print(Z[i])
        k = select_K[i]
        color_mtx[i, :] = np.array(color[k]) / 255.0
    """pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
    pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)"""

    if iteration == 0:

        pcl = trimesh.points.PointCloud(pts_mtx, colors=[0.1, 0.1, 0.1])
        scene = trimesh.Scene()
        scene.add_geometry(pcl)
        scene.camera_transform = transform
        #scene.show(line_settings={'point_size':10})
        png = scene.save_image(resolution=[640, 480], visible=True)
        file_name = 'source_seg.png'
        with open(file_name, 'wb') as f:
            f.write(png)
            f.close()

        pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
        scene = trimesh.Scene()
        scene.add_geometry(pcl)
        scene.camera_transform = transform
        #scene.show(line_settings={'point_size':10})
        png = scene.save_image(resolution=[640, 480], visible=True)
        file_name = 'source.png'
        with open(file_name, 'wb') as f:
            f.write(png)
            f.close()

    for t in range(X.shape[0]):
        pts_mtx = np.copy(X[t])
        pcl = trimesh.points.PointCloud(pts_mtx, colors=[0.1, 0.1, 0.1])
        scene = trimesh.Scene()
        scene.add_geometry(pcl)
        scene.camera_transform = transform
        #scene.show(line_settings={'point_size':10})
        png = scene.save_image(resolution=[640, 480], visible=True)
        file_name = 'target_%d.png'%(t)
        with open(file_name, 'wb') as f:
            f.write(png)
            f.close()
    
    """if save:
        if data_dir is None:
            data_dir = "results/00010_frame/100_"
        print("save")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd_trans)
        vis.run()
        image = vis.capture_screen_float_buffer()
        #plt.imshow(np.asarray(image))
        #plt.show()
        plt.imsave(data_dir + "%05d.png"%(save_image_idx), np.asarray(image))
        vis.destroy_window()
        save_image_idx += 1
    else:
        print("visualize")
        o3d.visualization.draw_geometries([pcd_trans])"""
        
    #pts_mtx = np.zeros((nPoints, 3))
    #color_mtx = np.zeros((nPoints, 3))
    """for t in range(TY.shape[0]):
        pts_mtx = []
        color_mtx = []
        for i in range(nPoints):
            for k in range(Z.shape[1]):
                if Z[i][k] > 0.3:
                    #print(Z[i][k])
                    color_mtx.append(np.array(color[k]) / 255.0)
                    pts_mtx.append(TY[t, k, i, :].numpy())
        if len(pts_mtx) == 0:
            break
        color_mtx = np.asarray(color_mtx)
        pts_mtx = np.asarray(pts_mtx)
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X[t])
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_trans = o3d.geometry.PointCloud()
        pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
        pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)

        if save:
            print("save")
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)
            vis.add_geometry(pcd_trans)
            vis.run()
            image = vis.capture_screen_float_buffer()
            #plt.imshow(np.asarray(image))
            #plt.show()
            plt.imsave("results/00010_step/500_%05d.png"%(save_image_idx), np.asarray(image))
            vis.destroy_window()
            save_image_idx += 1
        else:
            print("visualize")
            o3d.visualization.draw_geometries([pcd, pcd_trans])"""
            
    for t in range(TY.shape[0]):
        for i in range(nPoints):
            k = select_K[i]
            color_mtx[i, :] = np.array(color[k]) / 255.0
            pts_mtx[i, :] = TY[t, k, i, :]

        pcl = trimesh.points.PointCloud(np.copy(Y), colors=color_mtx)
        scene = trimesh.Scene()
        scene.add_geometry(pcl)
        scene.camera_transform = transform
        #scene.show(line_settings={'point_size':10})
        png = scene.save_image(resolution=[640, 480], visible=True)
        file_name = 'seg/seg_%d_%d.png'%(t, iteration)
        with open(file_name, 'wb') as f:
            f.write(png)
            f.close()

        pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
        scene = trimesh.Scene()
        scene.add_geometry(pcl)
        scene.camera_transform = transform
        #scene.show(line_settings={'point_size':10})
        png = scene.save_image(resolution=[640, 480], visible=True)
        file_name = 'recon/recon_%d_%d.png'%(t, iteration)
        with open(file_name, 'wb') as f:
            f.write(png)
            f.close()

        """pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X[t])
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_trans = o3d.geometry.PointCloud()
        pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
        pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)

        if save:
            if data_dir is None:
                data_dir = "results/00010_frame/100_"
            print("save")
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)
            vis.add_geometry(pcd)
            vis.add_geometry(pcd_trans)
            vis.run()
            image = vis.capture_screen_float_buffer()
            #plt.imshow(np.asarray(image))
            #plt.show()
            plt.imsave(data_dir + "%05d.png"%(save_image_idx), np.asarray(image))
            vis.destroy_window()
            save_image_idx += 1
        else:
            print("visualize")
            o3d.visualization.draw_geometries([pcd, pcd_trans])"""


def visualize_multiple_images(iteration, error, X, Y, TY, Z, P=None, dist=None, save=False, data_dir=None):
    global save_image_idx
    select_K = np.argmax(Z, axis=1)
    nPoints = Y.shape[0]
    pts_mtx = np.copy(Y)
    color_mtx = np.zeros((nPoints, 3))

    from scipy.spatial.transform import Rotation as R

    camera_rotation = np.eye(4)
    rotation = R.from_euler('xyz', [90, 0, 60], degrees=True).as_matrix()
    camera_rotation[0:3, 0:3] = rotation
    camera = trimesh.scene.Camera(fov=(np.pi/4.0, np.pi/4.0))
    transform = camera.look_at([[0.0, 0.0, 0.0]], rotation=camera_rotation, distance=5.0)
    
    color = []
    for k in range(Z.shape[1]):
        color.append(255.0 * np.random.rand(3))
        
    color[0:14] = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],[0,0,128]]

    for i in range(nPoints):
        #print(Z[i])
        k = select_K[i]
        color_mtx[i, :] = np.array(color[k]) / 255.0
    """pcd_trans = o3d.geometry.PointCloud()
    pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
    pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)"""

    pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
    scene = trimesh.Scene()
    scene.add_geometry(pcl)
    scene.camera_transform = transform
    scene.show(line_settings={'point_size':10})
    
    """if save:
        if data_dir is None:
            data_dir = "results/00010_frame/100_"
        print("save")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pcd_trans)
        vis.run()
        image = vis.capture_screen_float_buffer()
        #plt.imshow(np.asarray(image))
        #plt.show()
        plt.imsave(data_dir + "%05d.png"%(save_image_idx), np.asarray(image))
        vis.destroy_window()
        save_image_idx += 1
    else:
        print("visualize")
        o3d.visualization.draw_geometries([pcd_trans])"""
        
    #pts_mtx = np.zeros((nPoints, 3))
    #color_mtx = np.zeros((nPoints, 3))
    """for t in range(TY.shape[0]):
        pts_mtx = []
        color_mtx = []
        for i in range(nPoints):
            for k in range(Z.shape[1]):
                if Z[i][k] > 0.3:
                    #print(Z[i][k])
                    color_mtx.append(np.array(color[k]) / 255.0)
                    pts_mtx.append(TY[t, k, i, :].numpy())
        if len(pts_mtx) == 0:
            break
        color_mtx = np.asarray(color_mtx)
        pts_mtx = np.asarray(pts_mtx)
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X[t])
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_trans = o3d.geometry.PointCloud()
        pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
        pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)

        if save:
            print("save")
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)
            vis.add_geometry(pcd_trans)
            vis.run()
            image = vis.capture_screen_float_buffer()
            #plt.imshow(np.asarray(image))
            #plt.show()
            plt.imsave("results/00010_step/500_%05d.png"%(save_image_idx), np.asarray(image))
            vis.destroy_window()
            save_image_idx += 1
        else:
            print("visualize")
            o3d.visualization.draw_geometries([pcd, pcd_trans])"""
            
    for t in range(TY.shape[0]):
        for i in range(nPoints):
            k = select_K[i]
            color_mtx[i, :] = np.array(color[k]) / 255.0
            pts_mtx[i, :] = TY[t, k, i, :]

        pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
        scene = trimesh.Scene()
        scene.add_geometry(pcl)
        scene.camera_transform = transform
        scene.show(line_settings={'point_size':10})

        pcl = trimesh.points.PointCloud(pts_mtx, colors=[0.1, 0.1, 0.1])
        scene = trimesh.Scene()
        scene.add_geometry(pcl)
        scene.camera_transform = transform
        scene.show(line_settings={'point_size':10})

            
        """pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X[t])
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_trans = o3d.geometry.PointCloud()
        pcd_trans.points = o3d.utility.Vector3dVector(pts_mtx)
        pcd_trans.colors = o3d.utility.Vector3dVector(color_mtx)

        if save:
            if data_dir is None:
                data_dir = "results/00010_frame/100_"
            print("save")
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=True)
            vis.add_geometry(pcd)
            vis.add_geometry(pcd_trans)
            vis.run()
            image = vis.capture_screen_float_buffer()
            #plt.imshow(np.asarray(image))
            #plt.show()
            plt.imsave(data_dir + "%05d.png"%(save_image_idx), np.asarray(image))
            vis.destroy_window()
            save_image_idx += 1
        else:
            print("visualize")
            o3d.visualization.draw_geometries([pcd, pcd_trans])"""


"""def visualize_multiple_images(iteration, error, X, Y, TY, Z, P=None, dist=None, save=False, data_dir=None):
    global save_image_idx
    select_K = np.argmax(Z, axis=1)
    nPoints = Y.shape[0]
    pts_mtx = np.copy(Y)
    color_mtx = np.zeros((nPoints, 3))
    
    color = []
    for k in range(Z.shape[1]):
        color.append(np.random.rand(3))
        
    for i in range(nPoints):
        k = select_K[i]
        color_mtx[i, :] = color[k]

    pcl = trimesh.points.PointCloud(pts_mtx, colors=color_mtx)
    scene = trimesh.Scene()
    scene.add_geometry(pcl)
    scene.show(line_settings={'point_size':10})"""


def visualize_3d_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 3D
    Input: 
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlim([-0.2, 0.2])
    axes.set_ylim([-0.2, 0.2])
    axes.set_zlim([-1, 1])
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title('3D GMM')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.view_init(35.246, 45)
    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/3D_GMM_demonstration.png', dpi=100, format='png')
    plt.show()


def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input: 
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax

def visualize_2D_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 2D
    Input: 
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''
    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        plt.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])
        for j in range(8):
            axes.add_patch(
                patches.Ellipse(mu[:, i], width=(j+1) * stdev[0, i], height=(j+1) *  stdev[1, i], fill=False, color=[0.0, 0.0, 1.0, 1.0/(0.5*j+1)]))
        plt.title('GMM')
    plt.xlabel('X')
    plt.ylabel('Y')

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/2D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()
