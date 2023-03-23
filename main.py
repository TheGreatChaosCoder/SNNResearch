import LeakyBrain as brain
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

def main():
    print("Starting Program...")
    
    num_steps = 50
    data_path='./data/mnist'

    #Resize to 48x48 pixel image, grayscale, convert to a normalized tensor
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

    print("Downloading Datasets...")
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    #print(mnist_train)

    print("Downloading Completed")
    snn = brain.Brain(mnist_train, mnist_test, 200, .5)

    print("Starting SNN...")
    _, _ = snn.forward_pass(num_steps)

    print("Getting Initial Model...")
    snn.get_model_parameters('./data/parameters')

    # print("Finding Accuracy...")
    # snn.batch_accuracy(num_steps)

    print("Training SNN...")
    snn.training_loop(num_steps, .9, .999, num_iters=324)

    print("Getting Final Model...")
    snn.get_model_parameters('./data/parameters')

    print("Ending Program...")

main()