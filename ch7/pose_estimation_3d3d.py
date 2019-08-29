import cv2 as cv
import numpy as np
import g2o
from ch7.triangulation import find_feature_matches, pixel2cam

K = np.array([[520.9, 0, 325.1],
              [0, 521.0, 249.7],
              [0, 0, 1]])


def pose_estimation_3d3d(p_1, p_2):
    q1 = p_1 - np.mean(p_1, axis=0)
    q2 = p_2 - np.mean(p_2, axis=0)
    W = sum([np.matmul(q1i[:, np.newaxis], q2i[:, np.newaxis].transpose()) for q1i, q2i in zip(q1, q2)])
    print('W = \n', W)
    U, _, V = np.linalg.svd(W)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        U[:, 2] *= -1
    print('U = \n', U)
    print('V = \n', V)
    r_mat = np.matmul(U, V.transpose())

    t_vec = np.mean(p_1, axis=0)[:, np.newaxis] - r_mat.dot(np.mean(p_2, axis=0)[:, np.newaxis])

    return r_mat, t_vec


def bundle_adjustment(ps_1, ps_2, r_mat, t_vec):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSim3(g2o.LinearSolverCSparseSim3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    pose = g2o.VertexSE3Expmap()
    pose.set_estimate(g2o.SE3Quat(np.identity(3), np.zeros((3,))))
    pose.set_id(0)
    optimizer.add_vertex(pose)

    index = 1
    for p1, p2 in zip(ps_1, ps_2):
        edge = g2o.EdgeStereoSE3ProjectXYZOnlyPose()
        edge.cam_project(p2)
        edge.set_id(index)
        edge.set_vertex(0, pose)
        edge.set_measurement(p1)
        edge.set_information(np.identity(3))
        optimizer.add_edge(edge)
        index += 1

    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(100)
    print('T = \n', pose.estimate().matrix())


if __name__ == '__main__':
    img_1 = cv.imread('1.png')
    img_2 = cv.imread('2.png')
    key_points_1, key_points_2, matches = find_feature_matches(img_1, img_2)
    print('一共找到了', len(matches), '组匹配点')

    depth1 = cv.imread('1_depth.png', -1)
    depth2 = cv.imread('2_depth.png', -1)
    p_3d_1 = []
    p_3d_2 = []
    for m in matches:
        d1 = depth1[int(key_points_1[m.queryIdx].pt[1]), int(key_points_1[m.queryIdx].pt[0])]
        d2 = depth2[int(key_points_2[m.trainIdx].pt[1]), int(key_points_2[m.trainIdx].pt[0])]
        if d1 == 0 or d2 == 0:
            continue
        d1 = d1/5000.0
        d2 = d2/5000.0
        p1 = pixel2cam(key_points_1[m.queryIdx].pt, K)
        p2 = pixel2cam(key_points_2[m.trainIdx].pt, K)
        p_3d_1.append([p1[0, 0]*d1, p1[1, 0]*d1, d1])
        p_3d_2.append([p2[0, 0]*d2, p2[1, 0]*d2, d2])

    p_3d_1 = np.array(p_3d_1)
    p_3d_2 = np.array(p_3d_2)
    print("3d-2d pairs: ", p_3d_1.shape[0])

    R, t = pose_estimation_3d3d(p_3d_1, p_3d_2)
    print('ICP via SVD results: ')
    print('R = \n', R, '\nt = \n', t)
    print('R_inv = \n', R.transpose(), '\nt_inv = \n', -R.transpose().dot(t))

    print('calling bundle adjustment')
    bundle_adjustment(p_3d_1, p_3d_2, R, t)

    for p1, p2 in zip(p_3d_1[:5, :], p_3d_2[:5, :]):
        print('p1 = ', p1)
        print('p2 = ', p2)
        print('R * p2 + t = ', R.dot(p2[:, np.newaxis]) + t)
