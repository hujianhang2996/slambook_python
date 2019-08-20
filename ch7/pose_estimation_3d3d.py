import cv2 as cv
import numpy as np
import scipy.optimize as sco
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


def bundle_adjustment(p_1, p_2, r_mat, t_vec):
    def func(T, p1, p2):
        r = np.array(T[:9]).reshape((3, 3))
        t = np.array(T[9:]).reshape((3, 1))
        p1_ = (r.dot(p2.transpose()) + t).transpose()
        return np.linalg.norm(p1_ - p1, axis=1)

    T0 = np.concatenate((r_mat.flatten(), t_vec.flatten()))

    result = sco.leastsq(func, T0, args=(p_1, p_2))
    return np.array(result[0][:9]).reshape((3, 3)), np.array(result[0][9:]).reshape((3, 1))


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

    R_1, t_1 = bundle_adjustment(p_3d_1, p_3d_2, R, t)
    print('ICP via BA results: ')
    print('R = \n', R_1, '\nt = \n', t_1)
    print('R_inv = \n', R_1.transpose(), '\nt_inv = \n', -R_1.transpose().dot(t_1))