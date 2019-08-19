# -*-coding:utf-8-*-

import cv2 as cv
import numpy as np

K = np.array([[520.9, 0, 325.1],
              [0, 521.0, 249.7],
              [0, 0, 1]])


def pose_estimation_2d2d(kp_1, kp_2, m):
    points_1 = np.array([kp_1[match.queryIdx].pt for match in m])
    points_2 = np.array([kp_2[match.trainIdx].pt for match in m])

    fundamental_matrix, _ = cv.findFundamentalMat(points_1, points_2, method=cv.FM_8POINT)
    print('fundamental matrix is: \n', fundamental_matrix)

    essential_matrix, _ = cv.findEssentialMat(points_1, points_2, K)
    print('essential matrix is: \n', essential_matrix)

    homography_matrix, _ = cv.findHomography(points_1, points_2, cv.RANSAC, 3)
    print('homography matrix is: \n', homography_matrix)

    _, r_m, t_vec, _ = cv.recoverPose(essential_matrix, points_1, points_2, K)
    print('R: \n', r_m)
    print('t: \n', t_vec)
    return r_m, t_vec, essential_matrix


def find_feature_matches(i_1, i_2):
    orb = cv.ORB_create()
    kp_1 = orb.detect(i_1)
    kp_2 = orb.detect(i_2)
    kp_1, des_1 = orb.compute(i_1, kp_1)
    kp_2, des_2 = orb.compute(i_2, kp_2)

    matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
    m_es = matcher.match(des_1, des_2)
    min_dist = np.min([match.distance for match in m_es])
    max_dist = np.max([match.distance for match in m_es])
    print('--Max dist: ', max_dist)
    print('--Min dist: ', min_dist)
    good_matches = [match for match in m_es if match.distance <= max(2*min_dist, 30)]
    return kp_1, kp_2, good_matches


def pixel2cam(p, c_m):
    return np.array([[(p[0] - c_m[0, 2]) / c_m[0, 0]], [(p[1] - c_m[1, 2]) / c_m[1, 1]]])


if __name__ == '__main__':
    img_1 = cv.imread('1.png')
    img_2 = cv.imread('2.png')
    key_points_1, key_points_2, matches = find_feature_matches(img_1, img_2)
    print('一共找到了', len(matches), '组匹配点')

    R, t, E = pose_estimation_2d2d(key_points_1, key_points_2, matches)

    # 验证E=t^R*scale
    t_x = np.array([[0, -t[2, 0], t[1, 0]],
                    [t[2, 0], 0, -t[0, 0]],
                    [-t[1, 0], t[0, 0], 0]])
    print('t^R = \n', np.matmul(t_x, R))
    print('E ./ t^R = \n', np.matmul(t_x, R) / E, '应该为一常数。')

    # 验证对极约束
    for m in matches:
        pt1 = pixel2cam(key_points_1[m.queryIdx].pt, K)
        y1 = np.concatenate((pt1, np.array([[1]])))
        pt2 = pixel2cam(key_points_1[m.trainIdx].pt, K)
        y2 = np.concatenate((pt2, np.array([[1]])))
        d = y2.transpose() * t_x * R * y1
        print('epipolar constraint = ', np.linalg.norm(d))
