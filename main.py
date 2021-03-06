import torch
import torch.backends.cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from dataset.YCG09dataset import YCG09DataSet
from dataset.ctpnDataset import CTPNDataset
from model.crnn.dense_full_conv import DenseNetCTC
from model.ctc import CTCLoss
from model.localization.ctpn.ctpn_loss import CTPNLoss
from utils.accuracy_fn import multi_label_accuracy_fn
from utils.label import MultiLabelTransformer
from utils.pytorch_trainer import Trainer


def train_ctpn(data_path, model_path=None, initial_lr=0.01, epochs=10, batch_size=32,
               load_worker=4, print_interval=10, gpu_id=-1, lr_decay_rate=2):
    import torchvision.models as models
    from model.localization.ctpn.ctpn import CTPN

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    model = CTPN(models.vgg16_bn(True).named_parameters())
    for name, parameter in model.named_parameters():
        if name.find('cnn') >= 0:
            parameter.requires_grad = False
    train_loader = torch.utils.data.DataLoader(
        CTPNDataset(data_path, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=load_worker, pin_memory=True
    )

    criterion = CTPNLoss(cuda=gpu_id >= 0)
    optimizer = torch.optim.SGD(model.parameters(), initial_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    trainer = Trainer(model=model, initial_lr=initial_lr, train_loader=train_loader, validation_loader=None,
                      top_k=(1,), print_interval=print_interval, loss_fn=criterion, optimizer=optimizer,
                      half_float=False, gpu_id=gpu_id, lr_decay_rate=lr_decay_rate, accuracy_fn=None,
                      model_store_path='/mnt/data/checkpoints/' + model.__class__.__name__)
    trainer.run(epochs=epochs, resume=model_path, valid=False)


def train_recognize(model, data_path, model_path=None, initial_lr=0.01, epochs=10, batch_size=32,
                    load_worker=4, start_index=0, print_interval=10, gpu_id=-1, lr_decay_rate=2,
                    train_length=None, valid_length=None):
    label_transformer = MultiLabelTransformer(label_file='label.txt', encoding='GB18030')
    start_index = 0 if start_index <= 0 else (start_index * batch_size) + 1
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(
        YCG09DataSet(data_path, True, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), data_length=train_length, start_index=start_index, split_char=' '),
        batch_size=batch_size, shuffle=False,
        num_workers=load_worker, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        YCG09DataSet(data_path, False, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), data_length=valid_length),
        batch_size=batch_size, shuffle=False,
        num_workers=load_worker, pin_memory=True
    )

    criterion = CTCLoss(use_baidu=True)
    optimizer = torch.optim.SGD(model.parameters(), initial_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    trainer = Trainer(model=model, initial_lr=initial_lr, train_loader=train_loader, validation_loader=val_loader,
                      top_k=(1,), print_interval=print_interval, loss_fn=criterion, optimizer=optimizer,
                      half_float=False, label_transformer=label_transformer,
                      model_store_path='/mnt/data/checkpoints/' + model.__class__.__name__,
                      accuracy_fn=multi_label_accuracy_fn, gpu_id=gpu_id, lr_decay_rate=lr_decay_rate)
    trainer.run(epochs=epochs, resume=model_path, valid=True)


def test_data(model, weight_path, label_transformer, batch_size=16):
    load_worker = 8
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
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

    error_sample = []
    for sample, target in train_loader:
        (sample, filename) = sample
        sample = sample.cuda()
        target = target.cuda()
        output = model(sample)
        labels = label_transformer.parse_target(target)
        for idx, row in enumerate(label_transformer.parse_prediction(output, to_string=True)):
            pred = ''.join(list(filter(lambda x: x != 0, row[0])))
            label = ''.join(labels[idx])
            if pred != label:
                error_sample.append((filename[idx], ' '.join([str(num.item()) for num in target[idx]])))
                print(idx, len(train_loader))

    with open('/mnt/data/error.txt', "a+") as error_file:
        for line in error_sample:
            error_file.write(line[0] + " " + line[1] + "\n")


def main():
    """
    data_path = "/mnt/other/dataset/mlt"
    model_path = ''
    train_ctpn(data_path, model_path, epochs=10, batch_size=1, print_interval=1, gpu_id=0,
               initial_lr=1e-4, lr_decay_rate=2)
    """
    classes = 5990
    batch_size = 16
    data_path = "/mnt/data/dataset/YCG09"
    model_path = '/mnt/data/checkpoints/DenseNetCTC/checkpoint-1-val_prec_0.983-loss_0.108.pth'
    model = DenseNetCTC(num_classes=classes)
    train_recognize(model, data_path, batch_size=batch_size, load_worker=8, gpu_id=0,
                    lr_decay_rate=1, model_path=model_path, start_index=36600, initial_lr=1e-4,
                    train_length=None, valid_length=None)

    # test_data(model, model_path, label_transformer, batch_size)


def train_attention():
    classes = 5992
    epochs = 3
    initial_lr = 0.1
    print_interval = 10
    lr_decay_rate = 1
    gpu_id = 0
    batch_size = 32
    load_worker = 8
    data_path = "/mnt/data/dataset/YCG09"
    model_path = '/mnt/data/checkpoints/Attention/checkpoint-0-val_prec_0.539-loss_2.428.pth'
    start_index = 41200 + 750
    train_length, valid_length = None, None

    from model.crnn.attention import Attention
    from model.crnn.attention_loss import AttentionLoss
    # 11  -> training label max_length
    # 256 -> hidden size， also embedding_size
    model = Attention(256, classes, 11)
    criterion = AttentionLoss()

    start_index = 0 if start_index <= 0 else (start_index * batch_size) + 1
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(
        YCG09DataSet(data_path, True, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), data_length=train_length, start_index=start_index, split_char=' '),
        batch_size=batch_size, shuffle=False,
        num_workers=load_worker, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        YCG09DataSet(data_path, False, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), data_length=valid_length),
        batch_size=batch_size, shuffle=False,
        num_workers=load_worker, pin_memory=True
    )

    optimizer = torch.optim.SGD(model.parameters(), initial_lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    trainer = Trainer(model=model, initial_lr=initial_lr, train_loader=train_loader, validation_loader=val_loader,
                      top_k=(1,), print_interval=print_interval, loss_fn=criterion, optimizer=optimizer,
                      half_float=False, model_store_path='/mnt/data/checkpoints/' + model.__class__.__name__,
                      gpu_id=gpu_id, lr_decay_rate=lr_decay_rate)
    trainer.run(epochs=epochs, resume=model_path, valid=True)


if __name__ == '__main__':
    train_attention()
