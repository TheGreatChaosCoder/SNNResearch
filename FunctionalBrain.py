import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

import pandas as pd
import random

import csv
import math
import os

class FunctionDataset(Dataset):
    def __init__(self, annotations_file, path):
        if annotations_file is None:
            annotations_file, length = self.create_annotations_file(path)
            
            self.data = pd.read_csv(annotations_file)
        else:
            self.data = pd.read_csv(annotations_file)
        self.path = path

        self.numerical_col = ['x','y']
        self.output = ['output']

        self.numerical_data = self.data.drop(columns=self.output)
        self.numerical_data = torch.tensor(self.numerical_data.to_numpy()).float()
        self.output_data = self.data[self.output]
        self.output_data = torch.tensor(self.output_data.to_numpy().reshape(-1)).long()

    def create_annotations_file(self, path, num_steps = 200):
        length = 0
        file_name = f'{path}/train-labels.csv'

        with open(file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['x','y','output'])
            x_val = np.arange(0, 200.2, 0.2) #200.5 because it goes up to, excluding 200.5
            y_val = np.arange(0, 200.2, 0.2) 
            function = lambda a,b : math.sqrt(a**2 + b**2)

            for x in x_val:
                for y in y_val:
                    val = function(x,y)
                    
                    if val >= 50 and val <= 200:
                        output = 0

                        if(val == 200.0):
                            output = 14
                        elif(val < 60.0):
                            output = 0
                        else:
                            num = 1
                            for i in np.arange(60, 200, 10):
                                if(val >= i and val<i+10.0):
                                    output = num
                                    break
                                num += 1

                        # x_signal = torch.ones(num_steps) * (x/200.0)
                        # x_signal = torch.bernoulli(x_signal)

                        # y_signal = torch.ones(num_steps) * (y/200.0)
                        # y_signal = torch.bernoulli(y_signal)

                        writer.writerow([x,y, output])
                        length += 1

        return file_name, length

    def __len__(self):
        return len(self.output_data)
    
    def __getitem__(self, idx):
        return self.numerical_data[idx], self.output_data[idx]
    
    def get_input_size(self):
        return len(self.numerical_col)
    
    def get_output_size(self):
        #return len(self.output)
        return 15
    
    def get_data(self):
        return self.data
    
    def get_train_and_test_data(self, percent_test : float):

        total_records = self.__len__()
        test_records = int(total_records * percent_test)
        records_zip = list(zip(self.numerical_data, self.output_data))
        random.shuffle(records_zip)

        train_zip =  records_zip[:total_records-test_records]
        test_zip = records_zip[total_records-test_records:total_records]
        return train_zip, test_zip
    
    def get_dtype(self):
        return self.labels.dtypes

class FunctionalBrain ():
    def __init__(self, dataset : FunctionDataset, beta, batch_size = 1000):
        dtype = torch.float
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        spike_grad = surrogate.fast_sigmoid(slope=25)

        all_layers = []
        input_size = dataset.get_input_size()
        layers = [200, 150, 100, 50] #neurons at each layer

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], dataset.get_output_size()))
        all_layers.append(snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True))

        self.net = nn.Sequential(
                    *all_layers
                    ).to(self.device)

        self.train_zip, self.test_zip = dataset.get_train_and_test_data(0.2)

        self.train = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        #Init loss function
        self.loss_fn  = SF.ce_rate_loss()
    
    def forward_pass(self, num_steps, data = None, targets = None):
        if data is None:
            data, targets = zip(*self.train_zip)
            data = torch.stack(data)
            targets = torch.stack(targets)
            data = data.to(self.device)
            targets = targets.to(self.device)

        mem_rec = []
        spk_rec = []
        utils.reset(self.net)

        for step in range(num_steps):
            spk_out, mem_out = self.net(data)
            #print(f"Finished Forward Pass") if step+1==num_steps else None
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)

    def batch_accuracy(self, num_steps, loader = None) -> float:
        if loader is None:
            loader = self.train

        with torch.no_grad():
            total = 0
            acc = 0
            self.net.eval()
            
            train_loader = iter(loader)
            for data, targets in train_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                spk_rec, _ = self.forward_pass(num_steps, data)

                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)

        print(f"The total accuracy on the test set is: {acc/total * 100:.2f}%")
        return acc/total
    
    def training_loop(self, num_steps, beta1, beta2, num_epochs = 1, num_iters = 100) -> None:
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2, betas = (beta1, beta2))
        loss_hist = []
        test_acc_hist = []

        # Outer training loop
        for epoch in range(num_epochs):
            # Training loop
            for i, (data, targets) in enumerate(iter(self.train)):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # forward pass
                self.net.train()
                spk_rec, _ = self.forward_pass(num_steps, data, targets)

                # initialize the loss & sum over time
                loss_val = self.loss_fn(spk_rec, targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                print(f"At {i} iteration")
                # Test set
                if i % 25 == 0 and i>0:
                    with torch.no_grad():
                        self.net.eval()

                        # Test set forward pass
                        test_acc = self.batch_accuracy(num_steps, self.test)
                        print(f"Iteration {i}, Test Acc: {test_acc * 100:.2f}%\n")
                        test_acc_hist.append(test_acc.item())

                if i == num_iters:
                    break

        with torch.no_grad():
            self.net.eval()

            # Test set forward pass
            test_acc = self.batch_accuracy(num_steps, self.test)
            print(f"Final Iteration, Test Acc: {test_acc * 100:.2f}%\n")
            test_acc_hist.append(test_acc.item())