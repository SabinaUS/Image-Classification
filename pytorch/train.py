import sys
sys.path.append("..")
import numpy as np
import os
import torch
import torch.nn as nn
from resnet import *
from cifar10 import *
from logger import *

gpu_number = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
    
class Trainer():
    def __init__(self, train_batch_size=256, test_batch_size=500, exp_name='exp'):
        self.model = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()
        self.dataset = Cifar10(train_batch_size, test_batch_size, data_format='NCHW')
        self.image_shape = self.dataset.input_shape
        self.num_classes = self.dataset.num_classes
        self.logger = Logger(exp_dir=exp_name)
        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.1,
                                         momentum=0.9,
                                         weight_decay=1e-4)

    def train(self, epochs=1):
        self.model.cuda()
        for i in range(epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            self.dataset.shuffle_dataset()
            for j in range(self.dataset.train_batch_count):
                batch_images, batch_labels = self.dataset.next_aug_train_batch(j)
                batch_images = torch.tensor(batch_images).cuda()
                batch_labels = torch.tensor(batch_labels).cuda()
                self.model.train()
                logits = self.model(batch_images)
                loss = self.criterion(logits, batch_labels)
                total_loss += loss
                total_accuracy += self.accuracy(logits, batch_labels)[0][0]
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = total_loss / self.dataset.train_batch_count 
            avg_accuracy = total_accuracy / self.dataset.train_batch_count
            # logging training results
            self.logger.log('Training epoch {0}'.format(i))
            self.logger.log('    train loss {0}, train error {1}'.format(avg_loss, 1.0 - avg_accuracy))
            total_loss = 0.0
            total_accuracy = 0.0
            # evaluate on test set
            self.model.eval()
            for j in range(self.dataset.test_batch_count)  :
                batch_images, batch_labels = self.dataset.next_test_batch(j)
                batch_images = torch.tensor(batch_images).cuda()
                batch_labels = torch.tensor(batch_labels).cuda()
                logits = self.model(batch_images)
                loss = self.criterion(logits, batch_labels)
                total_loss += loss
                total_accuracy += self.accuracy(logits, batch_labels)[0][0]
            avg_loss = total_loss / self.dataset.test_batch_count
            avg_accuracy = total_accuracy / self.dataset.test_batch_count
            # logging validation results
            self.logger.log('    test loss {0}, test error {1}'.format(avg_loss, 1.0 - avg_accuracy))
            # save model
            save_model_file = os.path.join(self.logger.exp_dir, 'ResNet-model_epoch' + str(i))
            if i % 20 == 0:
                state = {'epoch': i + 1,
                         'state_dict': self.model.state_dict(),
                         'optimizer' : self.optimizer.state_dict()
                        } 
                torch.save(state, save_model_file)

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_( 1.0 / batch_size))
        return res


if __name__ == "__main__":
    trainer = Trainer(exp_name='exp2')
    trainer.train(epochs=400)
