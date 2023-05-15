import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
from PIL import Image, ImageFilter
from matplotlib import cm
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import *
import time
import matplotlib.pyplot as plt

from dataset_prep import MNISTDataset


class MNIST(Dataset):

    def __init__(self, ref, transform=None, **kwargs):
        super(MNIST, self).__init__()
        self.__transform = transform

        mnist = ref

        if not ('train' in kwargs.keys()):
            kwargs['train'] = False
        if not ('test' in kwargs.keys()):
            kwargs['test'] = False

        self.__train = kwargs['train']
        self.__test = kwargs['test']

        self.__data = mnist.get_training_data() if (self.__train or not self.__test) else mnist.get_testing_data()

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, item):

        data = self.__data[item]

        image, label = data
        if self.__transform:
            image = self.__transform(image)

        return image, torch.tensor(label)


class ToTensor:

    def __call__(self, sample):
        image = sample
        width, height = image.shape
        return torch.from_numpy(image).view(1, width, height)


class GetDCT:

    def __init__(self, n__coef):
        self.__n = n__coef

    def __call__(self, img):
        image = dct(dct(img.T, norm='ortho').T, norm='ortho')
        image = image.flatten()[0:self.__n]
        return torch.from_numpy(image)


class GetEdge:

    def __call__(self, image):
        image = Image.fromarray(np.uint8(cm.gist_earth(image) * 255)).convert("L")
        image = image.filter(ImageFilter.FIND_EDGES)
        image = np.array(image) / 255.5
        return image.flatten().astype(np.float32)


'''
    Model 1 : FFNN 


'''


class FCNeuralNet(nn.Module):

    def __init__(self, input_size, n_classes=10, hidden_depth=1, hidden_count=15, ):
        super(FCNeuralNet, self).__init__()

        self.__stack = nn.Sequential(
            nn.Linear(input_size, hidden_count),
            nn.ReLU(),
            *([nn.Linear(hidden_count, hidden_count), nn.ReLU()] * hidden_depth),
            nn.Linear(hidden_count, n_classes)
        )

    def forward(self, x):
        x = self.__stack(x)
        return x


"""

    Model 2 : LetNet 

"""


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# Hyper-parameters

EPOCHS = 10
LR = 1e-2
NUM_CLASS = 10
BATCH_SIZE = 4
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_test(model, train_loader, test_loader, l):
    # Device

    loss = nn.CrossEntropyLoss()
    opt = op.Adam(model.parameters(), lr=LR)

    # Training Loop

    st = time.time()

    for epoch in range(EPOCHS):

        for idx, (img, target) in enumerate(train_loader):
            opt.zero_grad()

            logits = model(img)
            L = loss(logits, target)

            L.backward()

            opt.step()

    et = time.time()

    print(f"Training Time ::--> {round(et - st, 5)} sec")

    st = time.time()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        et = time.time()
        print(f"Testing Time ::--> {round(et - st, 5)} sec")
        acc = 100 * correct / total
        print(f'Accuracy of the network on the {l} test images: {acc} %')

        return acc



def train_cnn(dataset="MNIST"):
    ref = MNISTDataset(dataset)
    train_dataset = MNIST(ref, Compose([ToTensor(), Resize([32, 32])]), train=True)
    test_dataset = MNIST(ref, Compose([ToTensor(), Resize([32, 32])]), test=True)

    model = LeNet5(NUM_CLASS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_test(model, train_loader, test_loader, len(test_dataset))


def train_pca(hidden_count, ):
    dataset = "MNIST"
    ref = MNISTDataset(dataset)
    ref.apply_pca(490)
    train_dataset = MNIST(ref, train=True)
    test_dataset = MNIST(ref, test=True)

    model = FCNeuralNet(490, NUM_CLASS, hidden_count)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_test(model, train_loader, test_loader, len(test_dataset))


def train_edge(hidden_count, dataset="MNIST"):
    ref = MNISTDataset(dataset)
    train_dataset = MNIST(ref, Compose([GetEdge()]), train=True)
    test_dataset = MNIST(ref, Compose([GetEdge()]), test=True)

    model = FCNeuralNet(784, NUM_CLASS, hidden_count)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_test(model, train_loader, test_loader, len(test_dataset))


def train_dct(hidden_count, dataset="MNIST"):
    ref = MNISTDataset(dataset)
    train_dataset = MNIST(ref, Compose([GetDCT(200)]), train=True)
    test_dataset = MNIST(ref, Compose([GetDCT(200)]), test=True)

    model = FCNeuralNet(200, NUM_CLASS, hidden_count)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_test(model, train_loader, test_loader, len(test_dataset))


if __name__ == '__main__':
    '''
    print("-------- DCT ----------")
    for i in range(3):
        print(f"DCT with number of hidden Layers ::-> {i + 1}")
        train_dct(i + 1)
        print('---------------------------------')

    print("\n\n\n------------ PCA ------------")
    for i in range(3):
        print(f"PCA with number of hidden Layers ::-> {i + 1}")
        train_pca(i + 1)
        print('---------------------------------')

    print("\n\n\n------------ Edge ------------")
    for i in range(3):
        print(f"Edge with number of hidden Layers ::-> {i + 1}")
        train_pca(i + 1)
        print('---------------------------------')
    '''

    print("CNN ------ MNIST")
    #train_cnn("MNIST")
    print("CNN ------ Speech")
    r = np.linspace(1e-5 , 0.1 , 20)
    print(r)
    acc_list = []
    for i in r:
        LR = i
        acc = train_cnn("S2S")
        acc_list.append(acc)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy(%)")
    plt.plot(r , acc_list)
    plt.show()

