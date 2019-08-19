import cv2 as cv
import numpy as np
# 读取图像
img_1 = cv.imread('1.png')
img_2 = cv.imread('2.png')
# 计算关键点
orb = cv.ORB_create()
kp_1 = orb.detect(img_1)
kp_2 = orb.detect(img_2)
# 计算关键点描述子
kp_1, des_1 = orb.compute(img_1, kp_1)
kp_2, des_2 = orb.compute(img_2, kp_2)
# 关键点匹配
matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
matches = matcher.match(des_1, des_2)
# 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
min_dist = np.min([match.distance for match in matches])
good_matches = [m for m in matches if m.distance < max(min_dist * 2, 30)]
# 绘制匹配结果
img_match = cv.drawMatches(img_1, kp_1, img_2, kp_2, matches, img_1)
img_good_match = cv.drawMatches(img_1, kp_1, img_2, kp_2, good_matches, img_1)
cv.imshow('所有匹配点对', img_match)
cv.imshow('优化后匹配点对', img_good_match)
cv.waitKey()
