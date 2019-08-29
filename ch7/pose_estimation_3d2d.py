import cv2 as cv
import numpy as np
import g2o
from ch7.triangulation import find_feature_matches, pixel2cam

K = np.array([[520.9, 0, 325.1],
              [0, 521.0, 249.7],
              [0, 0, 1]])


def bundle_adjustment(points_3d, points_2d, r_mat, t_vec):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    pose = g2o.VertexSE3Expmap()
    pose.set_estimate(g2o.SE3Quat(r_mat, t_vec.reshape((3, ))))
    pose.set_id(0)
    optimizer.add_vertex(pose)

    index = 1
    for p_3d in points_3d:
        point = g2o.VertexSBAPointXYZ()
        point.set_id(index)
        point.set_estimate(p_3d)
        point.set_marginalized(True)
        optimizer.add_vertex(point)
        index += 1

    camera = g2o.CameraParameters(K[0, 0], np.array([K[0, 2], K[1, 2]]), 0)
    camera.set_id(0)
    optimizer.add_parameter(camera)

    index = 1
    for p_2d in points_2d:
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_id(index)
        edge.set_vertex(0, optimizer.vertex(index))
        edge.set_vertex(1, pose)
        edge.set_measurement(p_2d)
        edge.set_parameter_id(0, 0)
        edge.set_information(np.identity(2))
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

    d1 = cv.imread('1_depth.png')
    pts_3d = []
    pts_2d = []
    for m in matches:
        d = d1[int(key_points_1[m.queryIdx].pt[1]), int(key_points_1[m.queryIdx].pt[0])][0]
        if d == 0:
            continue
        dd = d/5000.0
        p1 = pixel2cam(key_points_1[m.queryIdx].pt, K)
        pts_3d.append([p1[0, 0]*dd, p1[1, 0]*dd, dd])
        pts_2d.append(key_points_2[m.trainIdx].pt)

    pts_3d = np.array(pts_3d)
    pts_2d = np.array(pts_2d)
    print("3d-2d pairs: ", pts_3d.shape[0])

    _, r, t = cv.solvePnP(pts_3d, pts_2d, K, np.array([]))
    R = cv.Rodrigues(r)[0]
    print('R = \n', R, '\nt = \n', t)

    print('calling bundle adjustment')
    bundle_adjustment(pts_3d, pts_2d, R, t)
