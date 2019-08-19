import cv2
import numpy as np
import pcl


def get_transform(_p):
    q1, q2, q3, q0 = _p[3:]
    _R = np.array([[1-2*q2**2-2*q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
                  [2*q1*q2+2*q0*q3, 1-2*q1**2-2*q3**2, 2*q2*q3-2*q0*q1],
                  [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 1-2*q1**2-2*q2**2]])
    _t = _p[:3]
    return _R, _t


color_img_list = [cv2.imread('color/' + str(i+1) + '.png') for i in range(5)]
depth_img_list = [cv2.imread('depth/' + str(i+1) + '.pgm', -1) for i in range(5)]

poses = np.loadtxt('pose.txt')

cx = 325.5
cy = 253.5
fx = 518.0
fy = 519.0
depthScale = 1000.0

points = []

for i in range(5):
    print('processing img ', str(i), '..........')
    color_img = color_img_list[i]
    depth_img = depth_img_list[i]
    size = color_img.shape
    # 求解相机坐标到世界坐标的转换矩阵
    pose = poses[i]
    R, t = get_transform(pose)

    # 求解相机坐标系坐标
    row_mat = np.array([list(range(size[0]))] * size[1]).transpose()
    column_mat = np.array([list(range(size[1]))] * size[0])
    z_mat = depth_img / depthScale
    x_mat = np.multiply((column_mat - cx)/fx, z_mat)
    y_mat = np.multiply((row_mat - cy) / fy, z_mat)

    # 去除零点
    Pc_mat = np.stack((x_mat, y_mat, z_mat), axis=-1).reshape((-1, 3))
    rgb_mat = np.expand_dims((color_img[:, :, 2] * 16 ** 4 +
                              color_img[:, :, 1] * 16 ** 2 +
                              color_img[:, :, 0]).flatten(), axis=-1)
    zero_columns = np.where(Pc_mat[:, 2] == 0)[0]
    Pc_mat = np.delete(Pc_mat, zero_columns, 0)
    rgb_mat = np.delete(rgb_mat, zero_columns, 0)

    # 求解世界坐标系坐标
    Pw_mat = np.dot(R, Pc_mat.transpose()) + np.expand_dims(t, axis=-1)

    points.append(np.concatenate((Pw_mat.transpose(), rgb_mat), axis=-1))

points = np.concatenate(points)
p = pcl.PointCloud_PointXYZRGBA([[point[0],
                                  point[1],
                                  point[2],
                                  int(point[3])] for point in points])
pcl.save_XYZRGBA(p, 'map.pcd')
