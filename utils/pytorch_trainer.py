import math
import os
import shutil
import time
from threading import Timer

import torch
import torch.nn.utils
import torch.utils.data

from utils.accuracy_fn import default_accuracy_fn
from utils.label import DoNothingLabelTransformer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self, initial_lr=1e-3, model=None, loss_fn=None, optimizer=None,
                 train_loader=None, validation_loader=None, test_loader=None,
                 lr_decay_rate=5, model_store_path="checkpoints", gpu_id=-1,
                 label_transformer=DoNothingLabelTransformer(),
                 half_float=False, accuracy_fn=default_accuracy_fn, print_interval=1, top_k=(1,)):
        if gpu_id < 0:
            self.device = torch.device("cpu")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True

        self.lr = initial_lr
        self.lr_decay = lr_decay_rate
        self.half_float = half_float
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.label_transformer = label_transformer
        self.model = model.to(self.device) if not half_float else model.to(self.device).half()
        self.criterion = loss_fn.to(self.device)
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.print_interval = print_interval
        self.metric = dict()
        self.metric['train_loss'] = 0.
        self.metric['validation_loss'] = 0.
        self.model_path = model_store_path
        if self.model_path and not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if 1 not in top_k:
            top_k.append(1)
        self.top_k = top_k
        for K in top_k:
            self.metric['train_prec@top' + str(K)] = 0.
            self.metric['validation_prec@top' + str(K)] = 0.

        self.print_str = self._print_format()
        self.stopped = False

        def check_stop(trainer):
            if not os.path.exists('command.txt'):
                Timer(10.0, check_stop, args=[trainer], kwargs=None).start()
                return
            with open('command.txt', 'r') as order:
                line = order.readline()
                if line.find('stop') >= 0:
                    trainer.stopped = True
                else:
                    Timer(10.0, check_stop, args=[trainer], kwargs=None).start()

        Timer(10.0, check_stop, args=[self], kwargs=None).start()

    def _print_format(self):
        print_str = 'Epoch[{0}]: [{1}/{2}]\t' \
                    'ETA {3:.2f}\t' \
                    'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t' \
                    'Data {data_time.val:.3f}({data_time.avg:.3f})\t' \
                    'Loss {loss.val:.4f}({loss.avg:.4f})\t'
        if self.accuracy_fn is None:
            return print_str
        for k in self.top_k:
            print_str = print_str + 'Prec@' + str(k) + ' {metric[top' + str(k) + '].val:.2%}({metric[top' + str(
                k) + '].avg:.2%})\t'
        return print_str

    def _adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1 ** (epoch // self.lr_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("*" * 10, '==> lr =', lr, "*" * 10)

    def _save_checkpoint(self, epoch, is_best, file_prefix='checkpoint'):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_prec@top1': self.metric['train_prec@top1'],
            'validation_prec@top1': self.metric['validation_prec@top1'],
            'train_loss': self.metric['train_loss'],
            'validation_loss': self.metric['validation_loss'],
        }
        filename = '{filePrefix}-{epoch}-val_prec_{precTop1:.3f}-loss_{loss:.3f}.pth' \
            .format(filePrefix=file_prefix, epoch=epoch,
                    precTop1=state['validation_prec@top1'],
                    loss=state['validation_loss'])
        file_path = os.path.join(self.model_path, filename)
        torch.save(state, file_path)
        if is_best:
            shutil.copyfile(file_path, os.path.join(self.model_path, 'model_best.pth.tar'))
        print(">> model saved >>", file_path)

    def run(self, epochs=5, resume=None, valid=True):
        epoch = 0
        if resume is not None and os.path.exists(resume):
            checkpoint = torch.load(resume)
            epoch = checkpoint['epoch'] if self.train_loader.dataset.start_index > 0 else checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        best_prec = 0
        for idx in range(epoch, epochs):
            print('*' * 10, "epoch[", idx, "] training start", "*" * 10)
            self._do_run(True, idx)
            print('*' * 10, "epoch[", idx, "] training finished", "*" * 10)
            if valid:
                print('*' * 10, "epoch[", idx, "] validation start", "*" * 10)
                self._do_run(False, idx)
                print('*' * 10, "epoch[", idx, "] validation finished", "*" * 10)
            is_best = self.metric['validation_prec@top1'] > best_prec
            if is_best:
                best_prec = self.metric['validation_prec@top1']
            self._save_checkpoint(idx, is_best)
            if self.stopped:
                break
            # second epoch, reset index to  0
            self.train_loader.dataset.reset_index(False)

    def _do_run(self, train_model, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        prec_topk = dict()
        for k in self.top_k:
            prec_topk['top' + str(k)] = AverageMeter()

        # switch mode
        if train_model:
            self.model.train()
            self._adjust_learning_rate(self.optimizer, epoch)
        else:
            self.model.eval()

        end = time.time()
        loader = self.train_loader if train_model else self.validation_loader
        if train_model:
            self._compute(batch_time, data_time, end, epoch, loader, losses, prec_topk, train_model)
        else:
            with torch.no_grad():
                self._compute(batch_time, data_time, end, epoch, loader, losses, prec_topk, train_model)

        for k in self.top_k:
            if train_model:
                self.metric['train_prec@top' + str(k)] = prec_topk['top' + str(k)].avg
            else:
                self.metric['validation_prec@top' + str(k)] = prec_topk['top' + str(k)].avg
        if train_model:
            self.metric['train_loss'] = losses.avg
        else:
            self.metric['validation_loss'] = losses.avg

    def _compute(self, batch_time, data_time, end, epoch, loader, losses, prec_topk, train_model):
        for i, (samples, target) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)
            samples = samples.to(self.device) if not self.half_float else samples.to(self.device).half()
            if isinstance(target, torch.Tensor):
                target = target.to(self.device)

            # compute output and loss
            output = self.model(samples, target=target)
            loss = self.criterion(output, target)
            if math.isinf(loss.item()):
                continue
            losses.update(loss.item(), samples.size(0))

            # measure accuracy and record loss
            if self.accuracy_fn is not None:
                precs = self.accuracy_fn(self.label_transformer, self.top_k, output, target)
                for idx, key in enumerate(prec_topk):
                    prec_topk[key].update(precs[idx].item() if isinstance(precs[idx], torch.Tensor) else precs[idx],
                                          samples.size(0))

            # compute gradient and do SGD step
            if train_model:
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i != 0 and i % self.print_interval == 0:
                left_data = len(loader) - i
                print(self.print_str.format(
                    epoch, i, len(loader), left_data * batch_time.avg,
                    batch_time=batch_time, data_time=data_time, loss=losses, metric=prec_topk), flush=True)
                if i == 0:
                    batch_time.reset()
                data_time.reset()

                if self.stopped:
                    break
