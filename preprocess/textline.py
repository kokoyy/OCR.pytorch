import cv2
import numpy as np


def _preprocess(gray):
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilation2 = cv2.dilate(erosion, element3, iterations=2)
    return dilation2


def _find_text_region(image):
    region = []
    binary, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        if height > width * 1.2:
            continue

        region.append(box)

    return region


def cv2_detect_text_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilation = _preprocess(gray)
    return _find_text_region(dilation)


if __name__ == '__main__':
    img = cv2.imread('/home/yuanyi/Pictures/tijian.png', cv2.IMREAD_COLOR)
    region = cv2_detect_text_region(img)
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv2.imwrite("contours.png", img)
