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

class SNNDataset(Dataset):
    def __init__(self, annotations_file, path, increment = 0.5, normalize = True):
        if annotations_file is None:
            annotations_file, length = self.create_annotations_file(path)
            
            self.data = pd.read_csv(annotations_file)
        else:
            self.data = pd.read_csv(annotations_file)
        self.path = path

        self.numerical_col = ['P','Rant', 'f',]
        self.output = ['Pmax']

        self.numerical_data = self.data.drop(columns=self.output)
        self.numerical_data = torch.tensor(self.numerical_data.to_numpy()).float()
        self.output_data = self.data[self.output]
        self.output_data = torch.tensor(self.output_data.to_numpy().reshape(-1)).long()

    def create_annotations_file(self, path):
        length = 0
        file_name = f'{path}/mppt-data.csv'

        with open(file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['P','Rant', 'Lant', 'f', 'Effmax', 'Rrec', 'Lrec', 'Crec', 'Pmax'])
            P = np.arange(1.0, 1000.0 + 1, 1) #in microwatts
            Rant = np.arange(50, 125 + 1, 1) #in ohms
            Lant = np.arange(1.0, 1000.0 + 1, 1) #in microhenries
            freq = np.arange(.2, 2 + .2, .2) #in gigahertz
            Rrec =  np.arange(50, 125 + 1, 1) #in ohms
            Lrec = np.arange(1.0, 100.0 + 0.1, 0.1) #in microhenries
            Crec = np.arange(10, 10000.0 + 1, 1) #in nanofarads

            gamma = lambda L,C, w : L*w - 1/(C*w)
            ang_freq = lambda f : f*2*math.pi
            efficency = lambda Rant, Zant, Rrec, Zrec : 1 - (math.hypot((Rrec-Rant)**2,(Zrec-Zant)**2)/math.hypot((Rrec+Rant)**2,(Zrec+Zant)**2))**2

            max_vals = [0,0,0,0] #max value for eff at input val given Rec, Lrec, and Crec
            length = 0

            for f in freq:
                for p in P:
                    for rant in Rant:
                        for lant in Lant:
                            for rrec in Rrec:
                                for lrec in Lrec:
                                    for crec in Crec:
                                        w = ang_freq(f)
                                        Zant = lant * w
                                        Zrec = gamma(lrec, crec, w)
                                        eff = efficency(rant, Zant, rrec, Zrec)

                                        if max_vals[0] < eff:
                                            max_vals = [eff, rrec, lrec, crec]

                            writer.writerow([p, rant, lant, f, max_vals[0], max_vals[1], max_vals[2], max_vals[3], p*max_vals[0]])
                            length += 1
                            print(length)
                            max_vals = [0,0,0,0]

        return file_name, length

    def __len__(self):
        return len(self.output_data)
    
    def __getitem__(self, idx):
        return self.numerical_data[idx], self.output_data[idx]
    
    def get_input_size(self):
        #return len(self.numerical_col)
        return 2
    
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
    
dataset = SNNDataset(None, './data/MPPTData', increment=0.25, normalize = True)