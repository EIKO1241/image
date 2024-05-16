import cv2
import numpy as np
import getopt
import sys
import random

# 绘制不同颜色的匹配项和可选的一组内框
def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # 拼接两张图片，形成一张新图片
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # 将第一张图片放在左边
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # 第二张图片放在右边
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # 将匹配的每一对点用圆圈标出并连线
    for mat in matches:

        # 获取每张图片的匹配关键点
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # 在关键点对处画出小圆
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # 将关键点用直线连起来
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

# 用SIFT算法寻找关键点
def findFeatures(img):
    print("Finding Features...")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img = cv2.drawKeypoints(img, keypoints, None)
    cv2.imwrite('sift_keypoints.png', img)

    return keypoints, descriptors

# 匹配给定关键点、描述符和图像列表的特征
def matchFeatures(kp1, kp2, desc1, desc2, img1, img2):
    print("Matching Features...")
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    matchImg = drawMatches(img1,kp1,img2,kp2,matches)
    cv2.imwrite('Matches.png', matchImg)
    return matches

#计算出对应四个点的单应性
def calculateHomography(correspondences):
    #循环处理对应关系并创建集合矩阵
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)
    u, s, v = np.linalg.svd(matrixA)
    h = np.reshape(v[8], (3, 3))
    h = (1/h.item(8)) * h
    return h

# 计算估计点与原始点之间的几何距离
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

# 运行RANSAC算法，随机创建单应性图
def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(1000):
        # 随机取四个点计算单应性
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        # 调用这些点上的单应性函数
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers

# 运行函数
def main():
    args, img_name = getopt.getopt(sys.argv[1:],'', ['threshold='])
    args = dict(args)

    estimation_thresh = args.get('--threshold')
    print("Estimation Threshold: ", estimation_thresh)
    if estimation_thresh is None:
        estimation_thresh = 0.60

    estimation_thresh = float(estimation_thresh)

    img1name = str(img_name[0])
    img2name = str(img_name[1])

    img1 = cv2.imread(img_name[0], 0)
    img2 = cv2.imread(img_name[1], 0)

    # 找到特征和关键点
    correspondenceList = []
    if img1 is not None and img2 is not None:
        kp1, desc1 = findFeatures(img1)
        kp2, desc2 = findFeatures(img2)
        print("Found keypoints in " + img1name + ": " + str(len(kp1)))
        print("Found keypoints in " + img2name + ": " + str(len(kp2)))
        keypoints = [kp1,kp2]
        matches = matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
        for match in matches:
            (x1, y1) = keypoints[0][match.queryIdx].pt
            (x2, y2) = keypoints[1][match.trainIdx].pt
            correspondenceList.append([x1, y1, x2, y2])

        corrs = np.matrix(correspondenceList)

        # 调用RANSAC算法
        finalH, inliers = ransac(corrs, estimation_thresh)
        print("Final homography: ", finalH)
        print("Final inliers count: ", len(inliers))

        matchImg = drawMatches(img1,kp1,img2,kp2,matches,inliers)
        cv2.imwrite('InlierMatches.png', matchImg)

        f = open('homography.txt', 'w')
        f.write("Final homography: \n" + str(finalH)+"\n")
        f.write("Final inliers count: " + str(len(inliers)))
        f.close()


if __name__ == "__main__":
    main()
