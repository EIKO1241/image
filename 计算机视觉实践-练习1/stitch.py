import cv2
import numpy as np
import argparse

class stitch:
    def process(self,paths,mode):
        path1,path2=paths
        img1=cv2.imread(path1)
        img2=cv2.imread(path2)
        kp1,feat1=self.get_points(img1)
        kp2,feat2=self.get_points(img2)
        retval=self.matchs(kp1,kp2,feat1,feat2)
        if mode=="level":
            relt = cv2.warpPerspective(img2, retval, (img1.shape[1] + img2.shape[1], img1.shape[0]))
            # 将图片B传入
            #relt=relt[0:img1.shape[0], 0:img1.shape[1]]
            relt[0:img1.shape[0], 0:img1.shape[1]] = img1
            # 计算混合渐变
            mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
            mask[:, img1.shape[1]-40:] = 255
            relt = cv2.seamlessClone(img1, relt, mask, (img1.shape[1], img1.shape[0]//2), cv2.NORMAL_CLONE)
            relt=self.trim_border(relt,mode)
        elif mode=="vertical":
            relt = cv2.warpPerspective(img2, retval, (img1.shape[1], img1.shape[0] + img2.shape[0]))
            # 将图片B传入
            relt[0:img1.shape[0], 0:img1.shape[1]] = img1
            # 计算混合渐变
            mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
            mask[img1.shape[0]-40:, :] = 255
            relt = cv2.seamlessClone(img1, relt, mask, (img1.shape[1]//2, img1.shape[0]), cv2.MIXED_CLONE)
            relt=self.trim_border(relt,mode)
        # 返回匹配结果
        return relt
        pass
    
    #提取图像的特征点
    def get_points(self,img):
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.SIFT_create()
        # 检测特征点并计算描述子
        kps, features = descriptor.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])
        return kps, features
        pass

    #特征点匹配及过滤
    def matchs(self,kp1,kp2,feat1,feat2):
        # 建立暴力匹配器
        matcher=cv2.FlannBasedMatcher()
        #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 使用KNN检测来自AB图的SIFT特征匹配
        rawMatches = matcher.knnMatch(feat1, feat2, 2)

        # 过滤
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kp1[i] for (_, i) in matches])
            ptsB = np.float32([kp2[i] for (i, _) in matches])

            # 计算霍夫变换矩阵以及掩码
            retval, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 3.0)
            return retval
        else:
            raise KeyError("两张图片不匹配")
        pass

    def trim_border(self,image,mode):
        # 将彩色图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 查找灰度图像的轮廓
        contours,_ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 获取最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        # 获取凸包
        hull = cv2.convexHull(largest_contour)
        # 获取凸包的顶点
        hull_points = []
        for point in hull:
            hull_points.append(point[0])
        # 计算包围轮廓的最小矩形
        bounding_rect = cv2.boundingRect(contours[0])
        x, y, w, h = bounding_rect
        #获取边角点
        rhpoint = None
        rbpoint = None
        if mode=="level":
            h = image.shape[0] - 1
            for point in hull_points:
                if rhpoint is None or (point[1] == 0 and (rhpoint[1] == 0 and point[0] > rhpoint[0])):
                    rhpoint = point
                if rbpoint is None or (point[1] == h and (rbpoint[1] != h and point[0] > rbpoint[0])):
                    rbpoint = point
            # 裁剪图像
            trimmed_image = image[y:y+h, 0:x+w]
            original_points=np.float32([[0,0],rhpoint,rbpoint,[0,h]])
            corrected_points=np.float32([[0, 0], [rbpoint[0], 0],rbpoint ,[0,h] ])
        else:
            w = image.shape[1]-1
            for point in hull_points:
                if rhpoint is None or (point[0] == 0 and (rhpoint[0] == 0 and point[1] > rhpoint[1])):
                    rhpoint = point
                if rbpoint is None or (point[0] == w and (rbpoint[0] != w and point[1] > rbpoint[1])):
                    rbpoint = point
            # 裁剪图像
            trimmed_image = image[0:y+h, 0:x+w]
            original_points=np.float32([[0,0],[w,0],rbpoint,rhpoint])
            corrected_points=np.float32([[0, 0],[w,0],rbpoint,[0,rhpoint[1]]])
        

        # 计算透视变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(original_points, corrected_points)
        # 应用透视变换
        corrected_image = cv2.warpPerspective(trimmed_image, perspective_matrix, (trimmed_image.shape[1], trimmed_image.shape[0]))
        return corrected_image



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f","--first", required=False,default=r"image\example1\1.jpg", help="path to the first image")
    ap.add_argument("-s","--second", required=False,default=r"image\example1\2.jpg", help="path to the second image")
    ap.add_argument("-m", "--mode", required=False, default="level", help="vertical or level")
    args = vars(ap.parse_args())
    stitcher=stitch()
    result=stitcher.process([args["first"],args["second"]],args["mode"])
    cv2.imwrite(r'out_put\output_image.jpg', result)
    result=cv2.resize(result,(result.shape[1]//2,result.shape[0]//2))
    cv2.imshow("Result",result)
    cv2.waitKey(0)
    pass