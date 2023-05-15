import glob
import numpy as np
import os
import os.path
from PIL import Image
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MNISTDataset:

    def __init__(self, path="MNIST"):
        self.__training_path = os.path.join(path, 'Reduced Trainging data')
        self.__testing_path = os.path.join(path, 'Reduced Testing data')

        self.__training_data = self.__read_data(self.__training_path)
        self.__testing_data = self.__read_data(self.__testing_path)

    @staticmethod
    def __read_data(target_path):
        target = []

        for num in range(10):
            paths = glob.glob(os.path.join(target_path, f"{num}\\*.jpg"))
            for path in paths:
                image = np.array(Image.open(path).convert("L")).astype(np.float32)/255.0
                target.append((image, num))

        return target

    def plot_random_images(self , x =2,y=2 , train=True):
        rand_samples = [random.sample(self.__training_data if train else self.__testing_data , y) for i in range(x)]
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.subplots(x , y , sharey=True ,sharex=True)

        for xx in range(x):
            for yy in range(y):
                sample , label = rand_samples[xx][yy]
                ax[xx , yy].set_title(f"Sample of {label}")
                ax[xx, yy].imshow(sample , cmap='gray')

        fig.show()

    def get_training_data(self, flatten=False):
        if flatten:
            return [(img.flatten() , label) for img , label in self.__training_data]
        else:
            return self.__testing_data

    def get_testing_data(self, flatten=False):
        if flatten:
            return [(img.flatten() , label) for img , label in self.__testing_data]
        else:
            return self.__testing_data

    def apply_pca(self , n_component):
        # Train Data
        X_train = np.array([x[0] for x in self.get_training_data(True)])
        # Test Data
        X_test = np.array([x[0] for x in self.get_testing_data(True)])

        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)

        pca = PCA(n_components=n_component)

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        self.__training_data = [(X_train[i] , self.__training_data[i][1]) for i in range(len(self.__training_data))]
        self.__testing_data = [(X_test[i], self.__testing_data[i][1]) for i in range(len(self.__testing_data))]




if __name__ == '__main__':
    m = MNISTDataset("LOL")
    m.plot_random_images(3 , 3 , False)
    train , test = m.get_training_data(flatten=True) , m.get_testing_data(flatten=True)
    print(train[0][0].shape)
    plt.show()
