import time

from PIL import ImageFilter
from matplotlib import cm
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from dataset_prep import *
from scipy.fftpack import dct

def train_kmeans(n_clusters, X_train, y_train , X_test , y_test):
    km = KMeans(
        n_clusters=n_clusters,
        n_init=10, max_iter=100,
        tol=1e-04, random_state=0
    )
    st = time.time()
    km.fit(X_train)
    et = time.time()

    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(n_clusters):
        index = np.where(km.labels_ == i, 1, 0)
        num = np.bincount(y_train[index == 1]).argmax()
        reference_labels[i] = num

    acc = check_accuracy_kmean(km , X_test , y_test , reference_labels)

    return km , acc ,round(et - st , 5)


def check_accuracy_kmean(model: KMeans, X_test: np.ndarray, y_test: np.ndarray, ref: dict):
    predict = model.predict(X_test).astype(np.int)
    predict = [ref[i] for i in predict]
    return accuracy_score(predict , y_test)




def train_svm(X_train , y_train , X_test , y_test , kernal="linear"):
    clf = svm.SVC(kernel=kernal)  # Linear Kernel

    # Train the model using the training sets
    st = time.time()
    clf.fit(X_train, y_train)
    et = time.time()
    # Predict the response for test dataset
    score = clf.score(X_test , y_test)

    return clf, score , round(et - st , 5)


def dct_features(X , n_coef):
    dct_list = []
    for img in X:
        s =  dct(dct(img.T, norm='ortho').T, norm='ortho')[0:n_coef]
        dct_list.append(s.flatten())

    return dct_list

def get_pca(X_train , X_test, n_component):
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    pca = PCA(n_components=n_component)

    pca_train_features = pca.fit_transform(X_train)
    pca_test_features = pca.transform(X_test)

    return pca_train_features , pca_test_features , sum(pca.explained_variance_ratio_)

def edge_detect(image):
    image = Image.fromarray(np.uint8(cm.gist_earth(image)*255)).convert("L")
    image = image.filter(ImageFilter.FIND_EDGES)
    image = np.array(image)/255.5
    return image.flatten()


def my_features(X_train , X_test ):
    return [edge_detect(i) for i in X_train] , [edge_detect(i) for i in X_test]

def plot_svm(svm_cls_list , titles , X , y):
    h = 0.02
    X ,_ , __= get_pca(X , [] , 2)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


    for i, clf in enumerate(svm_cls_list):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(1, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
