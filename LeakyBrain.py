import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

import csv
    
class Brain:
    def __init__(self, data_train, data_test, batch_size, beta):
        dtype = torch.float
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data = (data_train, data_test)

        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.net = nn.Sequential(nn.Conv2d(1, 128, 7),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(128, 256, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(256, 512, 3),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(512*2*2, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(self.device)
        
        self.batch_size = batch_size

        self.train = DataLoader(self.data[0], batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(self.data[1], batch_size=batch_size, shuffle=True, drop_last=True)

        #Init loss function
        self.loss_fn  = SF.ce_rate_loss()

    def forward_pass(self, num_steps, data = None):
        if data is None:
            data, targets = next(iter(self.train))
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
                spk_rec, _ = self.forward_pass(num_steps, data)

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

    #Writes to a csv file
    def get_model_parameters(self, path):
        model_parts = ['conv1', 'leaky1', 'conv2', 'leaky2', 'conv3', 'leaky3', 'linear', 'leaky4']
        
        params = list(self.net.parameters())

        with open(f'{path}/model_parameters.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

        #For i==0,2,4 -> write out convolution matrix list for each matrix
        #For i==odd number -> write out synoptic weights for neurons
        #For i==6 -> write out weights for linea
            for i in range(len(params)):
                if i != 0:
                    writer.writerow('')

                writer.writerow(model_parts[i:i+1])

                writer.writerow(params[i])

        return



