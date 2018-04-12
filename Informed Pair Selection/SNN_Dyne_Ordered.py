from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random

import tensorflow as tf
import seed_reader
import matrix_average

#Keras imports
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD
from keras import backend as K

from gensim.models import Doc2Vec
from sklearn import preprocessing

#For reproducability
np.random.seed(1337)
random.seed(1337)
tf.set_random_seed(1337)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
    
#Define Euclidean distance function
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis = 1, keepdims = True))
    
#Define the shape of the output of Euclidean distance
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
        
#Define the contrastive loss function (as from Hadsell et al [1].)
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    #return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
    
#Positive and negative pair creation - alternates between positive and negative pairs 
def create_random_pairs(x, y, n):
    pairs = []
    pairs_a = []
    pairs_b = []
    labels = []
    
    for i in range(n):
        rand_pos_pair = x[random.choice(np.flatnonzero(y == y[i]))]
        rand_neg_pair = x[random.choice(np.flatnonzero(y != y[i]))]
        pairs += [[x[i], rand_pos_pair]]
        pairs_a += [x[i]]
        pairs_b += [rand_pos_pair]
        pairs += [[x[i], rand_neg_pair]]
        pairs_a += [x[i]]
        pairs_b += [rand_neg_pair]
        labels += [1, 0]
        
    return np.array(pairs), np.array(labels), np.array(pairs_a), np.array(pairs_b)
    
#Create the base network to be shared (equal to feature extraction)
def create_base_network(input_dim):
    seq = Sequential()
    #Create a fully connected layer (Dense) with an input_dim number of inputs and a ReLU activation function
    seq.add(Dense(300, input_shape = (input_dim,), activation = 'relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(300, activation = 'relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(300, activation = 'relu'))
    return seq

#Compute classification accuracy with a fixed threshold on distances
def compute_accuracy(predictions, labels):
    return np.mean(labels == (predictions.ravel() < 0.5))

#Read the dataset from doc2vec model  
def read_dataset():
    model = Doc2Vec.load('./IMDB_Doc2Vec_Model/imdb300.d2v')
    
    train_arrays = np.zeros((25000, 300))
    train_labels = np.zeros(25000)

    for i in range(12500):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 1
        train_labels[12500 + i] = 0
    
    test_arrays = np.zeros((25000, 300))
    test_labels = np.zeros(25000)
    
    for i in range(12500):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays[i] = model.docvecs[prefix_test_pos]
        test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
        test_labels[i] = 1
        test_labels[12500 + i] = 0
    
    return train_arrays, train_labels, test_arrays, test_labels

#Select random sample from training set based upon the specified seed    
def random_sample(sample_size, seed_number, x, y):
    sample = []
    label = []
    seed_list = seed_reader.read_seed(sample_size, seed_number)
    for item in seed_list:
        if item == 250000:
            sample.append(x[59999])
            label.append(y[59999])
        else:
            sample.append(x[item])
            label.append(y[item])
    return np.array(sample), np.array(label)

#Create test pairs by pairing a test instance with a prototypical example (centroid) of a class    
def create_test_pairs(x_train, y_train, x_test, y_test):
    pairs = []
    labels = []
    centroids = matrix_average.matrix_mean(x_train, y_train)
    
    for i in range(len(y_test)):
        for key in centroids:
            pairs += [[x_test[i], centroids[key][0].tolist()]]
            if(key == y_test[i]):
                labels += [1]
            else:
                labels += [0]
                
    return np.array(pairs), np.array(labels)
    
#Euclidean distance for k-NN 
def get_euclidean_distance(instance1, instance2):
    return np.linalg.norm(np.array(instance1)-np.array(instance2))

#Get neighbours  of a given instance  
def get_neighbours(instance, dataset, n):
    return dataset[np.argsort(np.linalg.norm(dataset - instance, axis=1))[:n]]
          
  
def run(j):   
    #Split and shuffle the data between the train and test sets
    x_train, y_train, x_test, y_test = read_dataset()
    
    x_train = preprocessing.normalize(x_train)
    x_test = preprocessing.normalize(x_test)
    
    ran_x_train, ran_y_train = random_sample(2500, j, x_train, y_train)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    ran_x_train = ran_x_train.astype('float32')
    
    input_dim = 300
    nb_epoch = 5
    
    #Create training and test positive and negative pairs
    full_tr_pairs, full_tr_y, pairs_a, pairs_b = create_random_pairs(ran_x_train, ran_y_train, len(ran_x_train))
    
    te_pairs, te_y = create_test_pairs(ran_x_train, ran_y_train, x_test, y_test)
    
    ran_y_train_list = ran_y_train.tolist()
    
    full_tr_pairs_list = full_tr_pairs.tolist()
    
    #Neighbourhood complexity-based ordering
    pair_ranking = []
    
    for pc in range(len(full_tr_pairs) - 1):
        ratio_good = 1.0
        ratio_bad = 1.0
        ratio = 0.0
        pc_a = full_tr_pairs_list[pc][0]
        pc_b = full_tr_pairs_list[pc][1]
        neighbours_a = get_neighbours(pc_a, ran_x_train, 5)
        neighbours_b = get_neighbours(pc_b, ran_x_train, 5)
        for neighbour_a in neighbours_a:
            neighbour_a_label = ran_y_train[np.where(ran_x_train == neighbour_a)[0][0]]
            for neighbour_b in neighbours_b:
                neighbour_b_label = ran_y_train[np.where(ran_x_train == neighbour_b)[0][0]]
                if(neighbour_a_label == neighbour_b_label):
                    if(full_tr_y[pc] == 1):
                        ratio_good += 1
                    else:
                        ratio_bad  += 1
                else:
                    if(full_tr_y[pc] == 1):
                        ratio_bad += 1
                    else:
                        ratio_good  += 1
            ratio = ratio_good / ratio_bad
        pair_ranking += [[ratio, full_tr_pairs[pc], full_tr_y[pc]]]
        
    pair_ranking = sorted(pair_ranking, key = lambda x: x[0], reverse = True)
    
    random_tr_pairs = [pair_ranking[0][1]]
    random_tr_y = [pair_ranking[0][2]]
    
    p = 1
    for p in range(len(pair_ranking)):
        random_tr_pairs += [pair_ranking[p][1]]
        random_tr_y += [pair_ranking[p][2]]
        
    full_tr_pairs = np.array(random_tr_pairs)
    full_tr_y = np.array(random_tr_y)
    
    #exhaustive_tr_pairs, exhaustive_tr_y = create_exhaustive_pairs(exhaustive_x_train, exhaustive_y_train)
    
    #Network definition
    base_network = create_base_network(input_dim)
    
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    
    #Because we use the same instance 'base network' the weights of the network will 
    #be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape = eucl_dist_output_shape)([processed_a, processed_b])
    
    model = Model(input = [input_a, input_b], output = distance)
    
    #Train (Compile and Fit) the model
    rms = SGD(lr = 0.001, momentum=0.8)
    model.compile(loss = contrastive_loss, optimizer = rms)
    
    values = []
    
    for i in range(20):
        model.fit([full_tr_pairs[:, 0], full_tr_pairs[:, 1]], full_tr_y,
              validation_data = ([te_pairs[:, 0], te_pairs[:, 1]], te_y),
              batch_size = 16, nb_epoch = nb_epoch, verbose = 1, shuffle = False)
        
        pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        te_acc = compute_accuracy(pred, te_y)
        
        values += [te_acc * 100]
        
        #Create Explore set
        random_tr_pairs, random_tr_y, pairs_a, pairs_b = create_random_pairs(ran_x_train, ran_y_train, len(ran_x_train))
        
        #Get the new representation of data
        intermediate_layer_model_a = Model(input=input_a, output=model.layers[2].get_output_at(1))
        intermediate_layer_model_b = Model(input=input_b, output=model.layers[2].get_output_at(2))

        intermediate_output = intermediate_layer_model_a.predict(ran_x_train)
        intermediate_output_a = intermediate_layer_model_a.predict(pairs_a)
        intermediate_output_b = intermediate_layer_model_b.predict(pairs_b)
        
        #Update neighbourhood complexity-based ordering for self-paced learning
        pair_ranking = []
        
        for pc in range(len(random_tr_pairs) - 1):
            ratio_good = 1.0
            ratio_bad = 1.0
            ratio = 0.0
            pc_a = intermediate_output_a[pc]
            pc_b = intermediate_output_b[pc]
            neighbours_a = get_neighbours(pc_a, intermediate_output, 5)
            neighbours_b = get_neighbours(pc_b, intermediate_output, 5)
            for neighbour_a in neighbours_a:
                neighbour_a_label = ran_y_train_list[np.where(intermediate_output == neighbour_a)[0][0]]
                for neighbour_b in neighbours_b:
                    neighbour_b_label = ran_y_train_list[np.where(intermediate_output == neighbour_b)[0][0]]
                    if(neighbour_a_label == neighbour_b_label):
                        if(random_tr_y[pc] == 1):
                            ratio_good += 1
                        else:
                            ratio_bad  += 1
                    else:
                        if(random_tr_y[pc] == 1):
                            ratio_bad += 1
                        else:
                            ratio_good  += 1
            ratio = ratio_good / ratio_bad
            pair_ranking += [[ratio, random_tr_pairs[pc], random_tr_y[pc], pairs_a[pc], pairs_b[pc]]]
        
        pair_ranking = sorted(pair_ranking, key = lambda x: x[0], reverse = True)
    
        random_tr_pairs = [pair_ranking[0][1]]
        random_tr_y = [pair_ranking[0][2]]
        pairs_a = [pair_ranking[0][3]]
        pairs_b = [pair_ranking[0][4]]
    
        p = 1
        for p in range(len(pair_ranking)):
            random_tr_pairs += [pair_ranking[p][1]]
            random_tr_y += [pair_ranking[p][2]]
            pairs_a += [pair_ranking[p][3]]
            pairs_b += [pair_ranking[p][4]]
        
        full_tr_pairs = np.array(random_tr_pairs)
        full_tr_y = np.array(random_tr_y)
    
    file = open("Dyne_Ord/results" + str(j) + ".txt", "w")
    
    for value in values:
        s = str(value) + "\n"
        file.write(s)
        
    file.close()
    
j = 0
while(j<10):
    run(j)
    j += 1