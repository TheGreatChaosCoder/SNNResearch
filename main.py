import LeakyBrain as brain
import FunctionalBrain as func
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

def main():
    print("Starting Program...")
    
    num_steps = 50
    data_path='./data/mnist'

    snn = brain.Brain(data_path, 200, .5)

    print("Starting SNN...")
    _, _ = snn.forward_pass(num_steps)

    print("Getting Initial Model...")
    snn.get_model_parameters('./data/parameters')

    print("Training SNN...")
    snn.training_loop(num_steps, .9, .999, num_iters=324)

    print("Getting Final Model...")
    snn.get_model_parameters('./data/parameters')

    print("Ending Program...")

def main2():
    csv_path = './data/function/train-labels.csv'
    dataset = func.FunctionDataset(csv_path, './data/function')

    brain = func.FunctionalBrain(dataset, .5, 200)
    print("Training SNN...")
    brain.training_loop(100, .9, .999, num_iters=10000)
    print("Ending Program...")

main2()