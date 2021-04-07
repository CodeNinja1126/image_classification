import os
import argparse

import dataset
import model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference config')
    parser.add_argument('-a', required=True, type=str, help='model state dict address')
    parser.add_argument('-s', required=True, type=str, help='submisson file name')

    args = parser.parse_args()

    # load eval csv
    result = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    
    # load model
    eval_model = model.classificationModel().to(device)
    model_state_dict = torch.load(args.a)
    eval_model.load_state_dict(model_state_dict)

    eval_model.eval()

    # load dataset
    eval_dataset = dataset.ValidationSet(dataset.data_transform)
    eval_dataloader = DataLoader(eval_dataset, 
                                batch_size=10,
                                num_workers=4)

    # inference
    with torch.no_grad():
        for data in tqdm.tqdm(eval_dataloader):
            images, idx = data
            outputs = eval_model(images.to(device))
            predicted = torch.argmax(outputs.data, 1)
            result.loc[idx.tolist(), 'ans'] = predicted.tolist()
    
    # save submission
    result.to_csv(args.s, index=False)