from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from tensorflow import keras
import numpy as np
import sys
import time
import multiprocessing
import json
#from mpl_toolkits.mplot3d import Axes3D
from parsing import load_data

mode = 'test'
# mode = 'normal'

CORES = 1

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
        if parameters['r_method'] == 'pca':  # PCA is handled here while RFE is handled within the classifiers
            training, testing = pca_reduce(training, testing, parameters['n_components'])
            new_parameters['n_components'] = None  # tells the SVM function not to apply filtering
        score_sum += ml_test(training, testing, new_parameters)
    return score_sum / n_iterations


''' This is used for creating parallel processes '''
def wrapper(args):
    return k_fold_validation(args[0], args[1], args[2], args[3])


def optimize_knn(data, labels, parameters, plotting=False):
    n_dimensions = list(parameters['dimension_range'])
    n_neighbors = list(range(1, 20))
    z = []
    for n in n_neighbors:
        acc_list = []
        for d in n_dimensions:
            parameters = {'n_components': d, 'n_neighbors': n, 'r_method': 'pca'}
            inputs = [(data, labels, knn_test, parameters.copy())] * CORES
            pool = multiprocessing.Pool(CORES)
            results = pool.map(wrapper, inputs)
            total = np.mean(results)
            acc_list.append(total)
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
        pool = multiprocessing.Pool(CORES)
        inputs = [(data, labels, svm_test, parameters.copy())] * CORES
        results = pool.map(wrapper, inputs)
        total = np.mean(results)
        acc_list.append(total)
    if plotting:
        plt.plot(dimensions, acc_list)
        plt.xlabel('number of dimensions')
        plt.ylabel('classifier accuracy')
        plt.show()
    return dimensions[np.argmax(acc_list)], max(acc_list)


def optimize_nn(data, labels, parameters, iterations = 5, plotting = False):
    hidden_layer_counts = range(0, 3)
    dimensions = parameters['dimension_range']
    acc_array = []
    for d in dimensions:
        acc_list = []
        for c in hidden_layer_counts:
            parameters = {'ratio': 1.5, 'n_layers': c, 'r_method': 'pca', 'n_components': d}
            acc = k_fold_validation(data, labels, nn_test, parameters)
            acc_list.append(acc)
        acc_array.append(acc_list)
    max_arg = np.argmax(np.asarray(acc_array))
    optimal_dimensions = dimensions[int(max_arg/len(hidden_layer_counts))]
    optimal_layers = max_arg % len(hidden_layer_counts)
    return optimal_dimensions, optimal_layers, np.max(np.asarray(acc_array))


def optimize_nn_unfiltered(data, labels, parameters=None, iterations = 5, plotting = False):
    hidden_layer_counts = range(0, 3)
    acc_list = []
    for c in hidden_layer_counts:
        parameters = {'ratio': 1.5, 'n_layers': c, 'r_method': None}
        acc = k_fold_validation(data, labels, nn_test, parameters)
        acc_list.append(acc)
    max_arg = np.argmax(np.asarray(acc_list))
    optimal_layers = hidden_layer_counts[max_arg]
    return optimal_layers, np.max(np.asarray(acc_list))


def optimize_rf(data, labels, parameters, iterations=10, plotting=False):
    k = 0.25  # proportion of data used for testing
    dimensions = parameters['dimension_range']
    acc_list = []
    for d in dimensions:
        parameters['n_components'] = d
        pool = multiprocessing.Pool(CORES)
        inputs = [(data, labels, rf_test, parameters.copy())] * CORES
        results = pool.map(wrapper, inputs)
        total = np.mean(results)
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


if __name__ == '__main__':
    k = 0.25  # proportion of data used for testing
    # load parameters
    options = json.loads(open('parameters.json', 'r').read())
    # load data
    if mode == 'test':
        pos, neg = load_data('formatted_data/rats_ileum')
    else:
        pos, neg = load_data(sys.argv[1])
    data, labels = pd_to_data(pos, neg)
    # test classifiers
    with open('results.txt', 'w') as out_file:
        # NN unfiltered
        if options['nn_unfiltered']['use']:
            print('optimizing NN without filtering')
            start = time.time()
            nn_unfiltered_results = optimize_nn_unfiltered(data, labels)
            end = time.time()
            out_file.write('#neural network without dimensional reduction\n')
            out_file.write('Optimum at {} hidden layers with {}% accuracy\n'.format(nn_unfiltered_results[0], nn_unfiltered_results[1]*100))
            out_file.write('Completed in {} seconds\n\n'.format(end-start))
            out_file.flush()
        # NN with PCA
        if options['nn_pca']['use']:
            print('optimizing NN with PCA')
            start = time.time()
            min_dimension = options['nn_pca']['min_dimension']
            max_dimension = options['nn_pca']['max_dimension']
            step = options['nn_pca']['step']
            parameters = {'dimension_range': range(min_dimension, max_dimension, step)}
            nn_pca_results = optimize_nn(data, labels, parameters)
            end = time.time()
            out_file.write('#neural network with dimensional reduction via PCA\n')
            out_file.write('Optimum at {} dimensions and {} hidden layers with {}% accuracy\n'.format(nn_pca_results[0], nn_pca_results[1], nn_pca_results[2]*100))
            out_file.write('Completed in {} seconds\n\n'.format(end-start))
            out_file.flush()
        # SVM with PCA
        if options['svm_pca']['use']:
            print('optimizing SVM with PCA')
            start = time.time()
            min_dimension = options['svm_pca']['min_dimension']
            max_dimension = options['svm_pca']['max_dimension']
            step = options['svm_pca']['step']
            parameters = {'r_method': 'pca', 'dimension_range': range(min_dimension, max_dimension, step)}
            svm_pca_results = optimize_svm(data, labels, parameters, iterations=10)
            end = time.time()
            out_file.write('#support vector machine with dimensional reduction via PCA\n')
            out_file.write('Optimum at {} dimensions with {}% accuracy\n'.format(svm_pca_results[0], svm_pca_results[1]*100))
            out_file.write('Completed in {} seconds\n\n'.format(end - start))
            out_file.flush()
        # SVM with RFE
        if options['svm_rfe']['use']:
            print('optimizing SVM with RFE')
            start = time.time()
            min_dimension = options['svm_rfe']['min_dimension']
            max_dimension = options['svm_rfe']['max_dimension']
            step = options['svm_rfe']['step']
            parameters = {'r_method': 'rfe', 'dimension_range': range(min_dimension, max_dimension, step)}
            svm_rfe_results = optimize_svm(data, labels, parameters, iterations=10)
            end = time.time()
            out_file.write('#support vector machine with dimensional reduction via RFE\n')
            out_file.write('Optimum at {} dimensions with {}% accuracy\n'.format(svm_rfe_results[0], svm_rfe_results[1]*100))
            out_file.write('Completed in {} seconds\n\n'.format(end - start))
            out_file.flush()
        # random forest with RFE
        if options['rf_rfe']['use']:
            print('optimizing random forest with RFE')
            start = time.time()
            min_dimension = options['rf_rfe']['min_dimension']
            max_dimension = options['rf_rfe']['max_dimension']
            step = options['rf_rfe']['step']
            parameters = {'r_method': 'rfe', 'dimension_range': range(min_dimension, max_dimension, step)}
            rf_rfe_results = optimize_rf(data, labels, parameters, iterations=6)
            end = time.time()
            out_file.write('#random forest with dimensional reduction via RFE\n')
            out_file.write('Optimum at {} dimensions with {}% accuracy\n'.format(rf_rfe_results[0], rf_rfe_results[1]*100))
            out_file.write('Completed in {} seconds\n\n'.format(end - start))
            out_file.flush()
        # random forest with PCA
        if options['rf_pca']['use']:
            print('optimizing random forest with PCA')
            start = time.time()
            min_dimension = options['rf_pca']['min_dimension']
            max_dimension = options['rf_pca']['max_dimension']
            step = options['rf_pca']['step']
            parameters = {'r_method': 'pca', 'dimension_range': range(min_dimension, max_dimension, step)}
            rf_pca_results = optimize_rf(data, labels, parameters, iterations=10)
            end = time.time()
            out_file.write('#random forest with dimensional reduction via PCA\n')
            out_file.write('Optimum at {} dimensions with {}% accuracy\n'.format(rf_pca_results[0], rf_pca_results[1]*100))
            out_file.write('Completed in {} seconds\n\n'.format(end - start))
            out_file.flush()
        # random forest with no filtering
        if options['rf_unfiltered']['use']:
            print('testing random forest without dimensional reduction')
            start = time.time()
            rf_unfiltered_results = rf_no_filtering(data, labels)
            end = time.time()
            out_file.write('#random forest without filtering\n')
            out_file.write('{}% accuracy\n'.format(rf_unfiltered_results))
            out_file.write('Completed in {} seconds\n\n'.format(end - start))
            out_file.flush()
        # kNN with PCA
        if options['knn_pca']['use']:
            print('testing kNN with PCA')
            start = time.time()
            parameters = {'dimension_range': range(1, 50, 2)}
            knn_pca_results = optimize_knn(data, labels, parameters)
            end = time.time()
            out_file.write('#k-nearest neighbor with dimensional reduction via PCA\n')
            out_file.write('Optimum at {} dimensions and {} neighbors with {}% accuracy\n'.format(knn_pca_results[0], knn_pca_results[1], knn_pca_results[2]*100))
            out_file.write('Completed in {} seconds\n\n'.format(end - start))
            out_file.flush()