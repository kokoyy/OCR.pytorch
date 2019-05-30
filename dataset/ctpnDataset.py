import os

import cv2
import torch.utils.data as data

import model.ctpn.ctpn_anchor as anchor_util


class CTPNDataset(data.Dataset):
    def __init__(self, file_path, transform=None, label_transform=None):
        self.img_file_path = os.path.join(file_path, 'image')
        self.label_file_path = os.path.join(file_path, 'label')
        self.transform = transform
        self.label_transform = label_transform
        self.all_sample = []
        for root, dirs, files in os.walk(self.img_file_path):
            for filename in files:
                label_file = os.path.join(self.label_file_path, filename.split('.')[0] + ".txt")
                if not os.path.exists(label_file):
                    continue
                with open(label_file, 'r') as anchor_text:
                    anchors = anchor_text.readlines()
                    if len(anchors) == 0:
                        continue
                self.all_sample.append(filename)

    def reset_index(self, use_start_index):
        pass

    def __len__(self):
        return len(self.all_sample)

    def __getitem__(self, index):
        sample = self.all_sample[index]
        image = cv2.imread(os.path.join(self.img_file_path, sample), cv2.IMREAD_UNCHANGED)
        label_file = os.path.join(self.label_file_path, sample.split('.')[0] + ".txt")
        with open(label_file, 'r') as anchor_text:
            anchors = anchor_text.readlines()
            positive, negative, vertical_reg, side_refinement_reg = anchor_util.parse(image, anchors)
            if self.transform is not None:
                image = self.transform(image)
            return image, [positive, negative, vertical_reg, side_refinement_reg]
