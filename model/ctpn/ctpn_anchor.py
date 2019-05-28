import math
import random

import numpy as np

ANCHOR_HEIGHTS = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]
FIX_ANCHOR_WIDTH = 16
NUM_OF_SAMPLE = 128


def parse(image, origin_anchors):
    """
    :param image: input picture, shape like (H, W, 3)
    :param origin_anchors: text like   ["201,162,207,229",
                                        "208,162,223,229",
                                        "224,162,239,229"]
                           each line was a anchor box in image
    :return: positive: 与ground truth的IOU大于0.7就是positive sample
             negative: 与ground truth的IOU小于0.5就是negative sample
             vertical_reg:
             side_refinement_reg:
    """
    positive = []
    negative = []
    vertical_reg = []
    side_refinement_reg = []

    (height, width) = (np.array(image.shape[:2]) / 16)  # assert input.shape[:2] % 16 == [0, 0]
    prepared_anchors = get_all_prepared_anchors(height, width)
    ground_truth_anchors = get_all_gt_anchors(origin_anchors)
    pred_gt_iou = {}
    for key in prepared_anchors:
        prepared_anchors_pre_space = prepared_anchors[key]
        for k, prepared_anchor in enumerate(prepared_anchors_pre_space):
            iou_key = key + '-' + str(k)
            if iou_key not in pred_gt_iou:
                pred_gt_iou[iou_key] = []
            for gt_key in ground_truth_anchors:
                gt_anchor = ground_truth_anchors[gt_key]
                iou = _cal_iou(prepared_anchor, gt_anchor[0], gt_anchor[1], gt_anchor[2])
                pred_gt_iou[iou_key].append(iou)

    for iou_key in pred_gt_iou:
        ious = pred_gt_iou[iou_key]
        indices = iou_key.split('-')
        prepared_anchor = prepared_anchors[indices[0] + "-" + indices[1]][int(indices[2])]

        ground_truth_anchor_array = [ground_truth_anchors[gt_key] for gt_key in ground_truth_anchors]
        max_iou = max(ious)
        if max_iou < 0.4:
            negative.append(prepared_anchor)
        elif max_iou > 0.7:
            positive.append(prepared_anchor)
        else:
            for gt_index, iou in enumerate(ious):
                if iou < 0.5:
                    continue
                gt_anchor = ground_truth_anchors[gt_index]
                is_side, is_left = _is_side_anchor(gt_index, gt_anchor, ground_truth_anchor_array)
                yaxis = prepared_anchor[3]
                prepared_anchor_height = ANCHOR_HEIGHTS[prepared_anchor[2]]
                # see paper https://arxiv.org/pdf/1609.03605.pdf
                vc = (yaxis - gt_anchor[0][1]) / prepared_anchor_height
                vh = math.log10(gt_anchor[1] / prepared_anchor_height)
                vertical_reg.append((prepared_anchor[0], prepared_anchor[1], prepared_anchor[2], vc, vh))
                if is_side:
                    x_axis = gt_anchor[3][0 if is_left else 2]
                    side_refinement_reg.append((prepared_anchor[0], prepared_anchor[1], prepared_anchor[2],
                                                (x_axis - prepared_anchor[1] * (16 if is_left else 17)) / 16))

    positive = random.sample(positive, min(NUM_OF_SAMPLE, len(positive)))
    negative = random.sample(negative, min(NUM_OF_SAMPLE, len(negative)))
    return positive, negative, vertical_reg, side_refinement_reg


def get_all_gt_anchors(origin_anchors):
    ground_truth_anchors = {}
    for num, anchor in enumerate(origin_anchors):
        points = [int(point) for point in anchor.strip().split(',')]
        anchor_center = ((points[0] + points[2]) / 2, (points[1] + points[3]) / 2)
        anchor_height = points[3] - points[1]
        anchor_width = points[2] - points[0]
        ground_truth_anchors[num] = (anchor_center, anchor_height, anchor_width, points)
    return ground_truth_anchors


def get_all_prepared_anchors(height, width):
    prepared_anchors = {}
    for i in range(0, int(width)):
        for j in range(0, int(height)):
            key = "{j}-{i}".format(j=j, i=i)
            if key not in prepared_anchors:
                prepared_anchors[key] = []
            for k in range(0, len(ANCHOR_HEIGHTS)):
                center = (j * 16 + (j + 1) * 16) / 2
                prepared_anchor_height = ANCHOR_HEIGHTS[k]
                prepared_anchor_top = center - prepared_anchor_height / 2
                prepared_anchor_bottom = center + prepared_anchor_height / 2
                if prepared_anchor_top < 0 or prepared_anchor_bottom > height * 16:
                    # not a valid anchor, skip
                    continue
                prepared_anchors[key].append((j, i, k, center))
    return prepared_anchors


def _cal_iou(prepared_anchor, gt_center, gt_height, gt_width):
    """
    calculate iou between prepared anchor and ground truth anchor
    :param prepared_anchor: shape like (j, i, k, center)
    :param gt_center:
    :param gt_height:
    :param gt_width:
    :return:
    """
    prepared_anchor_height = ANCHOR_HEIGHTS[prepared_anchor[2]]
    prepared_anchor_width = FIX_ANCHOR_WIDTH
    prepared_anchor_left = prepared_anchor[1] * 16
    prepared_anchor_right = (prepared_anchor[1] + 1) * 16 - 1
    prepared_anchor_top = prepared_anchor[3] - prepared_anchor_height / 2
    prepared_anchor_bottom = prepared_anchor[3] + prepared_anchor_height / 2

    gt_anchor_top = gt_center[1] - gt_height / 2
    gt_anchor_bottom = gt_center[1] + gt_height / 2
    gt_anchor_left = gt_center[0] - gt_width / 2
    gt_anchor_right = gt_center[0] + gt_width / 2
    # 判断在Y轴上是否有重合，若无，则IOU为0
    if gt_anchor_top <= prepared_anchor_top and gt_anchor_bottom < prepared_anchor_top:
        return 0
    if prepared_anchor_top <= gt_anchor_top and prepared_anchor_bottom < gt_anchor_top:
        return 0
    # 判断在X轴上是否有重合，若无，则IOU为0
    if gt_anchor_left <= prepared_anchor_left and gt_anchor_right < prepared_anchor_left:
        return 0
    if prepared_anchor_left <= gt_anchor_left and prepared_anchor_right < gt_anchor_left:
        return 0

    iou_width = min(prepared_anchor_right, gt_anchor_right) - max(prepared_anchor_left, gt_anchor_left)
    iou_height = min(prepared_anchor_bottom, gt_anchor_bottom) - max(prepared_anchor_top, gt_anchor_top)
    iou_area = iou_width * iou_height
    total_area = prepared_anchor_height * prepared_anchor_width + gt_height * gt_width - iou_area
    iou = iou_area / total_area
    return iou


def _is_side_anchor(anchor_index, ground_truth_anchor, ground_truth_anchors):
    """
    check if anchor is on the left or right side of Bbox
    :param anchor_index: index of ground_truth_anchor in ground_truth_anchors
    :param ground_truth_anchor:
    :param ground_truth_anchors:
    :return:
    """
    if anchor_index == 0 or anchor_index == len(ground_truth_anchors) - 1:
        return True, anchor_index == 0

    previous_ground_truth_anchor = ground_truth_anchors[anchor_index - 1]
    distance = math.fabs(previous_ground_truth_anchor[3][2] - ground_truth_anchor[3][0])  # 与上一个anchor的水平距离
    if distance > 1:
        #  距离超过1，说明不属于同一个Bbox
        return True, True

    next_ground_truth_anchor = ground_truth_anchors[anchor_index + 1]
    distance = math.fabs(ground_truth_anchor[3][2] - next_ground_truth_anchor[3][0])  # 与下一个anchor的水平距离
    if distance > 1:
        #  距离超过1，说明不属于同一个Bbox
        return True, False
    return False, False
