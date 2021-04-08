import os
import argparse

import dataset
import model
from AdamP import AdamP

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='Set some train option')
    parser.add_argument('-b', default=10, type=int, help='batch size (default : 10)')
    parser.add_argument('-e', default=5, type=int, help='epoch number (default : 5)')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate (default : 0.001)')
    parser.add_argument('-c', default=None, type=str, help='checkpoint adress')
    parser.add_argument('-s', default=None, required=True, type=str, help='trained state dict name')

    args = parser.parse_args()

    # load dataset
    mask_dataset = dataset.MaskImageDataset(dataset.data_transform)
    data_loader = DataLoader(mask_dataset,
                        shuffle=True,
                        batch_size=args.b, 
                        num_workers=4)

    # load model
    test_model = model.classificationModel().to(device)
    test_model.train()

    if args.c:
        model_state_dict = torch.load(args.c)
        test_model.load_state_dict(model_state_dict)

    # set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamP(test_model.parameters(), lr=args.lr)

    # training
    print('start training!')
    for epoch in range(args.e):

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            
            inputs, labels = data
        
            optimizer.zero_grad()
            labels = torch.flatten(torch.argmax(labels, dim=1))
            
            outputs = test_model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if i % 100 == 99:

                # measure accuracy
                with torch.no_grad():
                    accuracy = 0

                    for _ in range(10):
                        test_model.eval()
                        eval_input, label = mask_dataset[np.random.randint(0,len(mask_dataset))]
                
                        if torch.argmax(test_model(eval_input.unsqueeze(0).to(device))) == \
                            torch.argmax(label):
                            accuracy +=1

                    test_model.train()

                print('[%d, %5d] loss: %.3f accuracy: %.1f' %
                     (epoch + 1, i + 1, running_loss / 100, accuracy / 10))
                running_loss = 0.0


    print('Finished Training')
    # save model state_dict
    torch.save(test_model.state_dict(), args.s)
