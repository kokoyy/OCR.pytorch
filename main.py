import cv2
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from dataset.YCG09dataset import YCG09DataSet
from model.ctc import CTCLoss
from model.denselstmctc import DenseBLSTMCTC
from model.densenetctc import DenseNetCTC
from preprocess.textline import cv2_detect_text_region
from utils.accuracy_fn import multi_label_accuracy_fn
from utils.label import MultiLabelTransformer
from utils.pytorch_trainer import Trainer


def train(model, data_path, label_transformer, model_path=None, initial_lr=0.01, epochs=10, batch_size=32,
          load_worker=4, start_index=0, print_interval=10, gpu_id=-1, lr_decay_rate=2):
    start_index = 0 if start_index <= 0 else (start_index * batch_size) + 1
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(
        YCG09DataSet(data_path, True, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), start_index=start_index),
        batch_size=batch_size, shuffle=False,
        num_workers=load_worker, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        YCG09DataSet(data_path, False, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=load_worker, pin_memory=True
    )

    criterion = CTCLoss()
    optimizer = torch.optim.SGD(model.parameters(), initial_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    trainer = Trainer(model=model, initial_lr=initial_lr, train_loader=train_loader, validation_loader=val_loader,
                      top_k=(1,), print_interval=print_interval, loss_fn=criterion, optimizer=optimizer,
                      half_float=False, label_transformer=label_transformer,
                      accuracy_fn=multi_label_accuracy_fn, gpu_id=gpu_id, lr_decay_rate=lr_decay_rate)
    trainer.run(epochs=epochs, resume=model_path)


def test_data():
    classes = 5990
    batch_size = 32
    load_worker = 8
    label_transformer = MultiLabelTransformer(label_file='label.txt', encoding='GB18030')
    model = DenseNetCTC(num_classes=classes, conv0=nn.Conv2d(3, 64, 3, 1, 1))
    checkpoint = torch.load('checkpoints/checkpoint-1-val_prec_0.989-loss_0.015.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    train_data_path = "/mnt/data/BaiduNetdiskDownload"
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(
        YCG09DataSet(train_data_path, False, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=load_worker, pin_memory=True
    )

    for sample, target in train_loader:
        output = model(sample)
        target = label_transformer.parse_target(target)
        for idx, row in enumerate(label_transformer.parse_prediction(output, to_string=True)):
            pred = ''.join(list(filter(lambda x: x != 0, row[0])))
            label = ''.join(target[idx])
            if pred != label:
                print(''.join(list(filter(lambda x: x != 0, row[0]))), '\t===>\t', ''.join(target[idx]))


def evaluate(model_path, classes, image_path):
    label_transformer = MultiLabelTransformer(label_file='label.txt', encoding='GB18030')
    model = DenseBLSTMCTC(num_classes=classes, conv0=nn.Conv2d(3, 64, 3, 1, 1))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    font = ImageFont.truetype('fangzheng_heiti.TTF', 20)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    pilimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pilimg)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    text_lines = cv2_detect_text_region(img)
    for text_line in text_lines:
        left_bottom = text_line[0]
        right_top = text_line[2]
        region = img[min(right_top[1], left_bottom[1]):max(right_top[1], left_bottom[1]),
                     min(left_bottom[0], right_top[0]):max(left_bottom[0], right_top[0])]
        dst_shape = (int((32 / region.shape[0]) * region.shape[1]), 32)
        sample = cv2.resize(region, dst_shape, interpolation=cv2.INTER_NEAREST)

        sample = transform(sample)
        output = model(sample.unsqueeze(0))
        for idx, row in enumerate(label_transformer.parse_prediction(output, to_string=True)):
            pred = ''.join(list(filter(lambda x: x != 0, row[0])))
            print(pred)
            draw.text(text_line[3], pred, (0, 255, 0), font=font)

    return cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)


def main():
    classes = 5990
    batch_size = 32
    start_index = 33140
    data_path = "/mnt/data/BaiduNetdiskDownload"
    model_path = 'checkpoints/checkpoint-1-val_prec_0.909-loss_0.117.pth.tar'
    # model = DenseNetCTC(num_classes=classes, conv0=nn.Conv2d(3, 64, 3, 1, 1))
    model = DenseBLSTMCTC(num_classes=classes, conv0=nn.Conv2d(3, 64, 3, 1, 1))
    label_transformer = MultiLabelTransformer(label_file='label.txt', encoding='GB18030')
    train(model, data_path, label_transformer, batch_size=batch_size, load_worker=16, initial_lr=1e-3,
          gpu_id=0, lr_decay_rate=1, model_path=model_path, start_index=start_index)


if __name__ == '__main__':
    main()
    # test_data()
    # new_image = evaluate('checkpoints/checkpoint-1-val_prec_0.926-loss_0.100.pth.tar',
    #                     5990, '/home/yuanyi/Pictures/tijian.png')
    # cv2.imwrite('new_image.png', new_image)
