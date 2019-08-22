import cv2 as cv
import numpy as np
import time

path_to_dataset = './data'
detector = cv.FastFeatureDetector_create()
associate_file = open(path_to_dataset + '/associate.txt')
line = associate_file.readline()
time_rgb, rgb_file, time_depth, depth_file = line.rstrip('\n').split(' ')

depth = cv.imread(path_to_dataset + "/" + depth_file, -1)
pre_color = cv.imread(path_to_dataset + "/" + rgb_file)
pre_kps = np.float32([kp.pt for kp in detector.detect(pre_color)])
index = 0

while True:
    line = associate_file.readline()
    if not line:
        break
    time_rgb, rgb_file, time_depth, depth_file = line.rstrip('\n').split(' ')
    color = cv.imread(path_to_dataset + "/" + rgb_file)
    depth = cv.imread(path_to_dataset + "/" + depth_file, -1)
    if color is None or depth is None:
        continue
    tic = time.time()
    next_kps, status, err = cv.calcOpticalFlowPyrLK(pre_color, color, pre_kps, None)
    toc = time.time()
    print('LK Flow use time：', toc - tic, ' seconds.')
    next_kps = next_kps[np.where(status != 0)[0], :]
    if len(next_kps) == 0:
        print('all keypoints are lost.')
        break
    print('tracked keypoints: ', next_kps.shape[0])
    img_show = np.copy(color)
    for kp in next_kps:
        cv.circle(img_show, (kp[0], kp[1]), 5, (0, 240, 0))
        cv.imshow("corners", img_show)
    cv.waitKey(0)
    pre_color = color
    pre_kps = next_kps
    # if index % 50 == 0 or len(pre_kps) == 0:
    #     sorted_kps = detector.detect(color)
    #     sorted_kps.sort(key=lambda x: x.response)
    #     pre_kps = np.float32([kp.pt for kp in sorted_kps])
    #     pre_kps = pre_kps[:min(pre_kps.shape[0], 500), :]
    index += 1
