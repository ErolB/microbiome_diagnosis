from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from tensorflow import keras
import numpy as np

#from mpl_toolkits.mplot3d import Axes3D

from parsing import load_data

class RandomForestClassifierRFE(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierRFE, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


''' converts Pandas data frames to normalized positive and negative data sets '''
def pd_to_data(pos_frame, neg_frame):
    pos_array = pos_frame.values
    neg_array = neg_frame.values
    # normalize by count number
    training_data = []
    training_labels = []
    n_samples = pos_array.shape[1]
    for i in range(n_samples):
        training_data.append(pos_array[:,i] / sum(pos_array[:,i]) * 1000)
        training_labels.append(1)
        training_data.append(neg_array[:,i] / sum(neg_array[:,i]) * 1000)
        training_labels.append(0)
    return np.array(training_data), np.array(training_labels)


''' Reduce data dimensionality with PCA method '''
def pca_reduce(training, testing, n_components):
    reduction_model = PCA(n_components=n_components)
    reduction_model.fit(training[0])
    transformed_training_data = reduction_model.transform(training[0])
    transformed_testing_data = reduction_model.transform(testing[0])
    training = (transformed_training_data, training[1])
    testing = (transformed_testing_data, testing[1])
    return training, testing


''' Test SVM classifier '''
def svm_test(training, testing, parameters):
    n_components = parameters['n_components']
    if n_components is None:
        model = SVC(kernel='linear')  # no RFE filtering if n_components is None
    else:
        classifier = SVC(kernel='linear')
        model = RFE(classifier, n_components)
    model.fit(training[0], training[1])
    diagnosis = model.predict(testing[0])
    score = 0
    for i in range(len(diagnosis)):
        if (diagnosis[i] > 0.5) and (testing[1][i] == 1):
            score += 1
        if (diagnosis[i] < 0.5) and (testing[1][i] == 0):
            score += 1
    return score / len(diagnosis)


''' Test kNN classifier '''
def knn_test(training, testing, parameters):
    n_neighbors = parameters['n_neighbors']
    model = KNN(n_neighbors=n_neighbors, algorithm='brute')
    model.fit(training[0], training[1])
    diagnosis = model.predict(testing[0])
    #print('knn shape')
    #print(training[0].shape)
    score = 0
    for i in range(len(diagnosis)):
        if diagnosis[i] == testing[1][i]:
            score += 1
    return score / len(diagnosis)


''' Test random forest classifier '''
def rf_test(training, testing, parameters):
    if parameters['n_components'] is None:
        model = RandomForestClassifierRFE()  # no RFE filtering if n_components is None
    else:
        classifier = RandomForestClassifierRFE()
        model = RFE(classifier, parameters['n_components'])
    model.fit(training[0], training[1])
    diagnosis = model.predict(testing[0])
    score = 0
    for i in range(len(diagnosis)):
        if diagnosis[i] == testing[1][i]:
            score += 1
    return score / len(diagnosis)


''' Test neural network classifier '''
def nn_test(training, testing, parameters):
    input_dim = len(training[0])
    n_layers = parameters['n_layers']  # the number of fully connected layers
    size_ratio = parameters['ratio']
    # build network
    classifier = keras.Sequential()
    classifier.add(keras.layers.Dense(input_dim))
    for i in range(n_layers):
        classifier.add(keras.layers.Dense(input_dim * size_ratio, activation='sigmoid'))
    classifier.add(keras.layers.Dense(2))
    classifier.compile(loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'],
                       optimizer='adam')
    classifier.fit(training[0], training[1], epochs=20,
                   callbacks=[keras.callbacks.EarlyStopping(patience=2, monitor='loss')])
    return classifier.evaluate(testing[0], testing[1])[1]


''' Evaluate classifier with k-fold cross-validation '''
def k_fold_validation(data, labels, ml_test, parameters, k=0.25):
    t = int(len(labels) * k)  # size of test set
    n_iterations = int(1 / k)
    score_sum = 0
    for i in range(n_iterations):
        training_data = np.concatenate((data[0:t * i, :], data[t * (i + 1):t * n_iterations, :]), axis=0)
        training_labels = np.concatenate((labels[0:t * i], labels[t * (i + 1):t * n_iterations]), axis=0)
        training = (training_data, training_labels)
        testing_data = data[t * i:t * (i + 1), :]
        testing_labels = labels[t * i:t * (i + 1)]
        testing = (testing_data, testing_labels)
        new_parameters = parameters.copy()
        if parameters['r_method'] == 'pca':
            training, testing = pca_reduce(training, testing, parameters['n_components'])
            new_parameters['n_components'] = None  # tells the SVM function not to apply filtering
        score_sum += ml_test(training, testing, new_parameters)
    return score_sum / n_iterations

def optimize_rf(data, labels, parameters, iterations=10, plotting=False):
    k = 0.25  # proportion of data used for testing
    dimensions = parameters['dimension_range']
    acc_list = []
    for d in dimensions:
        parameters['n_components'] = d
        total = 0
        for i in range(iterations):
            total += k_fold_validation(data, labels, rf_test, parameters.copy())
        total /= iterations
        acc_list.append(total)
    optimal_dimension = dimensions[np.argmax(acc_list)]
    if plotting:
        plt.plot(dimensions, acc_list)
        plt.xlabel('number of dimensions')
        plt.ylabel('classifier accuracy')
        plt.show()
    return optimal_dimension, max(acc_list)


def rf_no_filtering(data, labels, iterations=10):
    parameters = {'r_method': None, 'n_components': None}
    total = 0
    for i in range(iterations):
        total += k_fold_validation(data, labels, rf_test, parameters)
    return total / iterations


def optimize_knn(data, labels, parameters, plotting=False):
    k = 0.25  # proportion of data used for testing
    n_dimensions = list(parameters['dimension_range'])
    n_neighbors = list(range(1, 20))
    z = []
    for n in n_neighbors:
        acc_list = []
        for d in n_dimensions:
            parameters = {'n_components': d, 'n_neighbors': n, 'r_method': 'pca'}
            acc_list.append(k_fold_validation(data, labels, knn_test, parameters))
        z.append(acc_list)
    optimum = np.argmax(np.array(z))
    optimal_value = np.array(z).flatten()[optimum]
    optimal_dimension = n_dimensions[int(optimum%len(n_dimensions))]
    optimal_neighbors = n_neighbors[int(optimum / len(n_dimensions))]
    n_neighbors, n_dimensions = np.meshgrid(n_neighbors, n_dimensions)
    z = np.array(z).transpose()
    if plotting:
        ax = plt.axes(projection='3d')
        ax.plot_surface(n_neighbors, n_dimensions, z)
        ax.set_xlabel('number of neighbors')
        ax.set_ylabel('number of dimensions')
        plt.show()
    return optimal_dimension, optimal_neighbors, optimal_value

def optimize_svm(data, labels, parameters, iterations = 10, plotting=False):
    k = 0.25  # proportion of data used for testing
    dimensions = parameters['dimension_range']
    acc_list = []
    for d in dimensions:
        parameters['n_components'] = d
        total = 0
        for i in range(iterations):
            total += k_fold_validation(data, labels, svm_test, parameters)
        acc_list.append(total/iterations)
    if plotting:
        plt.plot(dimensions, acc_list)
        plt.xlabel('number of dimensions')
        plt.ylabel('classifier accuracy')
        plt.show()
    return dimensions[np.argmax(acc_list)], max(acc_list)


def optimize_nn(data, labels, parameters, iterations = 5, plotting = False):
    hidden_layer_counts = range(0, 2)
    dimensions = parameters['dimension_range']
    acc_array = []
    for d in dimensions:
        acc_list = []
        for c in hidden_layer_counts:
            parameters = {'ratio': 1.5, 'n_layers': c, 'r_method': 'pca', 'n_components': d}
            acc = k_fold_validation(data, labels, nn_test, parameters)
            acc_list.append(acc)
        acc_array.append(acc_list)
    return np.argmax(np.asarray(acc_array)), np.max(np.asarray(acc_array))


if __name__ == '__main__':
    k = 0.25  # proportion of data used for testing
    pos, neg = load_data('formatted_data/rats_colon')
    data, labels = pd_to_data(pos, neg)
    with open('results.txt', 'w') as out_file:
        # NN with PCA
        '''
        parameters = {'dimension_range': range(10, 80, 20)}
        out_file.write(str(optimize_nn(data, labels, parameters)))
        # SVM with PCA
        parameters = {'r_method': 'pca', 'dimension_range': range(2, 20, 2)}
        out_file.write(str(optimize_svm(data, labels, parameters, iterations=10)))
        '''
        # SVM with RFE
        parameters = {'r_method': 'rfe', 'dimension_range': range(2, 20, 2)}
        out_file.write(str(optimize_svm(data, labels, parameters, iterations=10, plotting=True)))
        # random forest with RFE
        parameters = {'r_method': 'rfe', 'dimension_range': range(len(data[0])-1000, len(data[0]), 100)}
        out_file.write(str(optimize_rf(data, labels, parameters, iterations=10, plotting=True)))
        # random forest with PCA
        parameters = {'r_method': 'pca', 'dimension_range': range(len(data[0]) - 1000, len(data[0]), 100)}
        print(str(optimize_rf(data, labels, parameters, iterations=10)))
        # random forest with not filtering
        print(rf_no_filtering(data, labels))
        parameters = {'dimension_range': range(1, 50, 2)}
        print(str(optimize_knn(data, labels, parameters, plotting=True)))