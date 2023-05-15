import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import *


def save_df(name , no_c , acc , t):
    df = pd.DataFrame({"K-Mean" : no_c , "Accuracy" : acc , "Processing" : t})

    df.to_csv(name+".csv" , index_label=False , index=False)


def eval_dct_features():
    number_of_clusters = np.arange(1 , 33 , 1)
    t_list = []
    km_acc_list = []

    dataset = MNISTDataset()
    train_data, test_data = dataset.get_training_data(flatten=False), dataset.get_testing_data(flatten=False)
    # Training Data
    X_train, y_train = np.array([x[0] for x in train_data]), np.array([y[1] for y in train_data])

    # Test Data
    X_test, y_test = np.array([x[0] for x in test_data]), np.array([y[1] for y in test_data])

    X_train , X_test = dct_features(X_train , 200) , dct_features(X_test , 200)
    for i in number_of_clusters:
        km, acc, t = train_kmeans(i, X_train, y_train, X_test, y_test)
        km_acc_list.append(acc)
        t_list.append(t)
    plt.figure()
    plt.xlabel("Number of Clusters")
    plt.ylabel("Accuracy")
    plt.title("DCT as an input Features")
    plt.plot(number_of_clusters , km_acc_list)
    plt.figure()
    plt.xlabel("Number of Clusters")
    plt.ylabel("Time for fitting (Seconds)")
    plt.title(f"DCT as an input Features")
    plt.plot(number_of_clusters , t_list)
    plt.show()
    save_df("DCT_Data" , number_of_clusters , km_acc_list , t_list)


def eval_edge_features():
    number_of_clusters = np.arange(1 , 33 , 1)
    t_list = []
    acc_list = []
    dataset = MNISTDataset()
    train_data, test_data = dataset.get_training_data(flatten=False), dataset.get_testing_data(flatten=False)
    # Training Data
    X_train, y_train = np.array([x[0] for x in train_data]), np.array([y[1] for y in train_data])

    # Test Data
    X_test, y_test = np.array([x[0] for x in test_data]), np.array([y[1] for y in test_data])

    X_train , X_test = my_features(X_train , X_test)
    for i in number_of_clusters:
        km, acc, t = train_kmeans(i, X_train, y_train, X_test, y_test)
        acc_list.append(acc)
        t_list.append(t)

    plt.xlabel("Number of Clusters")
    plt.ylabel("Accuracy")
    plt.title("Edges as an input Features")

    plt.plot(number_of_clusters , acc_list)
    plt.figure()
    plt.xlabel("Number of Clusters")
    plt.ylabel("Time for fitting (Seconds)")
    plt.title(f"Edge as an input Features")
    plt.plot(number_of_clusters , t_list)
    plt.show()
    save_df("Edge_Data" , number_of_clusters , acc_list , t_list)


def eval_pca_features():
    number_of_clusters = np.arange(1 , 33 , 1)
    t_list = []
    acc_list = []
    dataset = MNISTDataset()
    train_data, test_data = dataset.get_training_data(flatten=True), dataset.get_testing_data(flatten=True)
    # Training Data
    X_train, y_train = np.array([x[0] for x in train_data]), np.array([y[1] for y in train_data])

    # Test Data
    X_test, y_test = np.array([x[0] for x in test_data]), np.array([y[1] for y in test_data])

    X_train , X_test , variance = get_pca(X_train , X_test , 490)
    for i in number_of_clusters:
        km, acc, t = train_kmeans(i, X_train, y_train, X_test, y_test)
        acc_list.append(acc)
        t_list.append(t)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Accuracy")
    plt.title(f"PCA as an input Features with var = {round(variance , 3)}")

    plt.plot(number_of_clusters , acc_list)
    plt.figure()
    plt.xlabel("Number of Clusters")
    plt.ylabel("Time for fitting (Seconds)")
    plt.title(f"PCA as an input Features with var = {round(variance , 3)}")
    plt.plot(number_of_clusters , t_list)
    plt.show()

    save_df("PCA_Data" , number_of_clusters , acc_list , t_list)


def svm_pca_test():
    dataset = MNISTDataset()
    train_data, test_data = dataset.get_training_data(flatten=True), dataset.get_testing_data(flatten=True)
    # Training Data
    X_train, y_train = np.array([x[0] for x in train_data]), np.array([y[1] for y in train_data])
    # Test Data
    X_test, y_test = np.array([x[0] for x in test_data]), np.array([y[1] for y in test_data])

    X_train_pca, X_test_pca, variance = get_pca(X_train, X_test, 490)
    pca_linear , acc_linear , t_linear = train_svm(X_train_pca , y_train , X_test_pca ,y_test , kernal="linear")
    pca_rbf, acc_rbf, t_rbf = train_svm(X_train_pca, y_train, X_test_pca, y_test, kernal="rbf")

    print("PCA Linear :  Acc " ,acc_linear,"  Time :  " , t_linear)
    print("PCA RBF :   Acc ", acc_rbf, "  Time :  ", t_rbf)

    #plot_svm([pca_linear , pca_rbf] , ["Linear PCA Features SVM" , "RBF PCA Features SVM"] , X_train_pca , y_train)


def svm_dct_test():
    dataset = MNISTDataset()
    train_data, test_data = dataset.get_training_data(flatten=False), dataset.get_testing_data(flatten=False)
    # Training Data
    X_train, y_train = np.array([x[0] for x in train_data]), np.array([y[1] for y in train_data])
    # Test Data
    X_test, y_test = np.array([x[0] for x in test_data]), np.array([y[1] for y in test_data])

    X_train_dct, X_test_dct = dct_features(X_train , 200) , dct_features(X_test , 200)
    dct_linear , acc_linear , t_linear = train_svm(X_train_dct, y_train, X_test_dct, y_test, kernal="linear")
    dct_rbf, acc_rbf, t_rbf = train_svm(X_train_dct, y_train, X_test_dct, y_test, kernal="rbf")

    print("DCT Linear :  Acc " ,acc_linear,"  Time :  " , t_linear)
    print("DCT RBF :   Acc ", acc_rbf, "  Time :  ", t_rbf)


def svm_edge_test():
    dataset = MNISTDataset()
    train_data, test_data = dataset.get_training_data(flatten=False), dataset.get_testing_data(flatten=False)
    # Training Data
    X_train, y_train = np.array([x[0] for x in train_data]), np.array([y[1] for y in train_data])
    # Test Data
    X_test, y_test = np.array([x[0] for x in test_data]), np.array([y[1] for y in test_data])

    X_train_edge, X_test_edge = my_features(X_train, X_test)
    edge_linear , acc_linear , t_linear = train_svm(X_train_edge, y_train, X_test_edge, y_test, kernal="linear")
    edge_rbf, acc_rbf, t_rbf = train_svm(X_train_edge, y_train, X_test_edge, y_test, kernal="rbf")

    print("Edge Linear :  Acc " ,acc_linear,"  Time :  " , t_linear)
    print("Edge RBF :   Acc ", acc_rbf, "  Time :  ", t_rbf)




if __name__ == '__main__':
    svm_edge_test()
