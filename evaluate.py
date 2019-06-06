import cv2
import torch
import torchvision.transforms as transforms

from preprocess.textline import cv2_detect_text_region
from utils.label import MultiLabelTransformer


def evaluate_full(model, model_path, image_path):
    label_transformer = MultiLabelTransformer(label_file='label.txt', encoding='GB18030')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    text_lines = cv2_detect_text_region(img)
    for text_line in text_lines:
        left_bottom = text_line[0]
        right_top = text_line[2]
        region = img[min(right_top[1], left_bottom[1]):max(right_top[1], left_bottom[1]),
                 min(left_bottom[0], right_top[0]):max(left_bottom[0], right_top[0])]
        cv2.rectangle(img, tuple(text_line[3]), tuple(text_line[1]), (0, 0, 255), 1)
        dst_shape = (int((32 / region.shape[0]) * region.shape[1]), 32)
        sample = cv2.resize(region, dst_shape)
        sample = transform(sample)
        output = model(sample.unsqueeze(0))
        for idx, row in enumerate(label_transformer.parse_prediction(output, to_string=True)):
            pred = ''.join(list(filter(lambda x: x != 0, row[0])))
            print(pred)


def evaluate(model, model_path, image_path):
    label_transformer = MultiLabelTransformer(label_file='label.txt', encoding='GB18030')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (int(img.shape[1] * 32 / img.shape[0]), 32), cv2.INTER_AREA)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    sample = transform(img)
    output = model(sample.unsqueeze(0))
    output = [output]
    print(label_transformer.parse_target(output))

    target = '89 201 241 178 19 94 19 22 26 656'
    target = [int(item) + 2 for item in target.split(' ')]
    print(label_transformer.parse_target([target]))


def main():
    classes = 5990
    # model = DenseNetLinear(num_classes=classes,
    #                       conv0=nn.Conv2d(3, 64, 3, 1, 1), avg_pool=True)
    from model.crnn.dense_full_conv import DenseNetCTC
    model = DenseNetCTC(classes)
    image = '/home/yuanyi/Pictures/tijian-photo2.jpg'

    # model = DenseNetCTC(num_classes=classes)
    evaluate_full(model, '/mnt/data/checkpoints/DenseNetCTC/checkpoint-2-val_prec_0.977-loss_0.011.pth',
                  image)


if __name__ == '__main__':
    main()
