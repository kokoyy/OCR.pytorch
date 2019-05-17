import torch.utils.data as data
import torch
import os
import cv2


class VGGDataSet(data.Dataset):
    def __init__(self, file_path, train_data=True, transform=None, label_transform=None, start_index=0):
        super(VGGDataSet, self).__init__()
        self.img_file_path = os.path.join(file_path, 'images')
        self.label_file = os.path.join(file_path, 'train.txt' if train_data else 'test.txt')
        self.transform = transform
        self.label_transform = label_transform
        self.all_sample = []
        self.start_index = start_index
        self.use_start_index = start_index > 0

        with open(self.label_file, 'r') as label_text:
            label_lines = label_text.readlines()
            for label_line in label_lines:
                labels = label_line.strip().split(' ')
                filename = labels[0]
                labels = [int(label.strip()) for label in labels[1:]]
                self.all_sample.append((filename, tuple(labels)))

    def reset_index(self, use_start_index):
        self.use_start_index = use_start_index

    def __len__(self):
        if self.use_start_index:
            return len(self.all_sample) - self.start_index
        else:
            return len(self.all_sample)

    def __getitem__(self, index):
        sample = self.all_sample[index + (self.start_index if self.use_start_index else 0)]
        image = cv2.imread(os.path.join(self.img_file_path, sample[0]), cv2.IMREAD_COLOR)
        if self.transform is not None:
            image = self.transform(image)
        target = torch.LongTensor(sample[1])
        if self.label_transform is not None:
            target = self.label_transform(target)
        return image, target


if __name__ == '__main__':
    dataset = VGGDataSet('/mnt/data/BaiduNetdiskDownload', False)
    for data in dataset:
        print(data)
