import cv2 as cv
import numpy as np
import scipy.optimize as sco
from ch7.triangulation import find_feature_matches, pixel2cam

K = np.array([[520.9, 0, 325.1],
              [0, 521.0, 249.7],
              [0, 0, 1]])


def bundle_adjustment(points_3d, points_2d, K, r_mat, t_vec):
    def func(T, p_2d, p_3d):
        r = np.array(T[:9]).reshape((3, 3))
        t = np.array(T[9:]).reshape((3, 1))
        p_2d_ = K.dot(r.dot(p_3d.transpose()) + t)
        p_2d_ = (p_2d_[:-1, :] / p_2d_[-1, :]).transpose()
        return np.linalg.norm(p_2d_ - p_2d, axis=1)

    T0 = np.concatenate((r_mat.flatten(), t_vec.flatten()))

    result = sco.leastsq(func, T0, args=(points_2d, points_3d))
    return np.array(result[0][:9]).reshape((3, 3)), np.array(result[0][9:]).reshape((3, 1))


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
    R_1, t_1 = bundle_adjustment(pts_3d, pts_2d, K, R, t)
    print('R = \n', R_1, '\nt = \n', t_1)
