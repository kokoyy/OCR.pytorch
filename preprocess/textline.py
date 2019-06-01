import cv2
import numpy as np


def get_projection_y(image):
    (h, w) = image.shape
    h_ = [0] * h
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    return h_


def get_projection_x(image):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.dilate(image, element, iterations=6)
    (h, w) = image.shape
    w_ = [0] * w
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    return w_


def _preprocess(gray):
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 12))
    dilation2 = cv2.dilate(erosion, element3, iterations=1)
    return dilation2


def _find_text_region_projection(image):
    position = []

    h = get_projection_y(image)
    start = 0
    h_start = []
    h_end = []
    for i in range(len(h)):
        if h[i] > 3 and start == 0:
            h_start.append(i)
            start = 1
        if h[i] <= 3 and start == 1:
            h_end.append(i)
            start = 0
    for i in range(len(h_start)):
        w = get_projection_x(image[h_start[i]:h_end[i], :])
        start = 0
        w_start = []
        w_end = []
        for k in range(len(w)):
            if w[k] > 1 and start == 0:
                w_start.append(k)
                start = 1
            if w[k] <= 1 and start == 1:
                w_end.append(k)
                start = 0

        for k in range(len(w_start)):
            position.append([(w_start[k], h_end[i] + 2), (w_end[k], h_end[i] + 2),
                             (w_end[k], h_start[i] - 2), (w_start[k], h_start[i] - 2)])
    return position


def _find_text_region_close(image):
    image = _preprocess(image)
    region = []
    binary, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < 100:
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return _find_text_region_projection(image)
