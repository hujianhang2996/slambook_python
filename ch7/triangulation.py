import cv2 as cv
import numpy as np
from ch7.pose_estimation_2d2d import find_feature_matches, pose_estimation_2d2d, pixel2cam

K = np.array([[520.9, 0, 325.1],
              [0, 521.0, 249.7],
              [0, 0, 1]])


def triangulation(kp_1, kp_2, ms, r_mat, t_vec):
    T1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    T2 = np.concatenate((r_mat, t_vec), axis=1)

    pts_1 = np.array([pixel2cam(kp_1[match.queryIdx].pt, K) for match in ms]).squeeze().transpose()
    pts_2 = np.array([pixel2cam(kp_2[match.trainIdx].pt, K) for match in ms]).squeeze().transpose()

    pts_4d = cv.triangulatePoints(T1, T2, pts_1, pts_2)
    points = pts_4d[:3, :] / pts_4d[3, :]
    return points.transpose()


if __name__ == '__main__':
    img_1 = cv.imread('1.png')
    img_2 = cv.imread('2.png')

    key_points_1, key_points_2, matches = find_feature_matches(img_1, img_2)
    print('一共找到了', len(matches), '组匹配点')
    R, t, E = pose_estimation_2d2d(key_points_1, key_points_2, matches)
    points = triangulation(key_points_1, key_points_2, matches, R, t)

    for match, point in zip(matches, points):
        print('-------------------------------------------------')
        pt1_cam = pixel2cam(key_points_1[match.queryIdx].pt, K)
        pt1_cam_3d = [point[0] / point[2], point[1] / point[2]]
        print('point in the first camera frame: ', pt1_cam.transpose().squeeze())
        print('point projected from 3D ', pt1_cam_3d, ', d=', point[2])

        pt2_cam = pixel2cam(key_points_2[match.trainIdx].pt, K)
        pt2_trans = np.matmul(R, point[:, np.newaxis]) + t
        pt2_trans = pt2_trans / pt2_trans[2, 0]
        print('point in the second camera frame: ', pt2_cam.transpose().squeeze())
        print('point reprojected from second frame: ', pt2_trans.transpose().squeeze())
