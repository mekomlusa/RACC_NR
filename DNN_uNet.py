# coding: utf-8
# An attempt to use UNet on the RACC_NR problem.

import tensorflow as tf
import keras
import numpy as np 
import keras.models
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose
from keras import backend as K
from math import log
#from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras import losses
from keras import regularizers
from models import Unet1D
import os


# Initial setup
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
K.set_session(tf.Session(config = config))

def shuffle_in_unison(a, b):
    """ Shuffling the data. """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def get_next_character(f):
    """Reads one character from the given textfile."""
    c = f.read(1)
    while c:
        yield c
        c = f.read(1)

def load_data(kind,label,fileNameCollection):
    """ Preparing the data. """
    x = []
    y = []
    
    if(kind == "train"):
        ground_turth_folder = "../4095_randomized/training/"
        noisy_folder = "../4095_noisy_randomized_0p01/training/"
    if(kind == "valid"):
        ground_turth_folder = "../4095_randomized/validation/"
        noisy_folder = "../4095_noisy_randomized_0p01/validation/"
    if(kind == "test"):
        ground_turth_folder = "../4095_randomized/testing/"
        noisy_folder = "../4095_noisy_randomized_0p01/testing/"
        
    groundTruths = os.listdir('%s' % ground_turth_folder+label)
    noises = os.listdir('%s' % noisy_folder+label)
    count = 0
    
    # fileNameCollection is a text file containing filenames of all the data (label and actual).
    # It is used here so that the program can read input data and its corresponding label in correct order.
    # also to remove inconsistent size concerns (e.g. having more labels than the features)
    with open(fileNameCollection, 'r') as f:
        alltext = f.read().strip().split('\n')
    
    for text in alltext:
        count += 1
        if count % 500 == 0:
            print("Loading "+kind+" example: "+str(count)+" for label : "+label)
            
        with open(ground_turth_folder+label+"/"+text, 'rb') as f:
            l = []
            for c in get_next_character(f):
                try:
                    l.append(int(c))
                except ValueError:
                    print("Error in ground truths:", c, text)
            l = np.array(l)
            y.append(l) 
        
        with open(noisy_folder+label+"/"+text, 'rb') as f:
            l = []
            for c in get_next_character(f):
                try:
                    l.append(int(c))
                except ValueError:
                    print("Error in noises:", c, text)
            l = np.array(l)
            x.append(l)
            
    return (np.array(x), np.array(y))

def negGrowthRateLoss(b,q):
    """ Customized loss function. """
    return (K.mean(-K.log(b +pow(-1,b)+pow(-1,b+1)*q)/K.log(2.0)))

def training(k, fileType, fileName, trainCollection, valCollection):
    """ Training the data."""
    input_rows, input_cols = k, 1
    #pad_dim = 32
    
    # the data, shuffled and split between train and test sets
    (x_train, y_train) = load_data("train", fileType, trainCollection)
    (x_valid, y_valid) = load_data("valid", fileType, valCollection)
    print('Before reshape:')
    print('x_train shape:', x_train.shape)
    print('x_valid shape:', x_valid.shape)
    # reshaping
    x_train = np.reshape(x_train,(len(x_train),input_rows,input_cols))
    x_valid = np.reshape(x_valid,(len(x_valid),input_rows,input_cols))
    y_train = np.reshape(y_train,(len(y_train),input_rows,input_cols))
    y_valid = np.reshape(y_valid,(len(y_valid),input_rows,input_cols))
    # x_train = np.repeat(x_train[:, :, np.newaxis], pad_dim, axis=2)
    # x_valid = np.repeat(x_valid[:, :, np.newaxis], pad_dim, axis=2)
    # y_train = np.repeat(y_train[:, :, np.newaxis], pad_dim, axis=2)
    # y_valid = np.repeat(y_valid[:, :, np.newaxis], pad_dim, axis=2)
    print('After reshape:')
    print('x_train shape:', x_train.shape)
    print('x_valid shape:', x_valid.shape)

    input_shape = (input_rows, input_cols, 1)

    # convert class vectors to binary class matrices
    print('Shuffling in unison')
    shuffle_in_unison(x_train,y_train)
    shuffle_in_unison(x_valid,y_valid)

    batch_size = 50
    epochs = 20

    # below is an example for the html U-Net model
    model = Unet1D(input_size=input_shape, k = k, loss=negGrowthRateLoss)
    filepath = "../results/"+fileName+".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger("../results/"+fileName+".csv")
    #plot_model(model,to_file="../results/"+fileName+".png",show_shapes =  True)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),callbacks=[checkpoint,csv_logger])
    model.save_weights(filepath)

def inference(k, persistance_path, fileType, output_file_name):
    """ Using the model presistence to do predictions."""
    input_rows, input_cols = k, 1
    pad_dim = 32
    
    # load the shuffled test data
    (x_test, y_test) = load_data("test",fileType, fileNameCollection)
    print('Before reshape:')
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    x_test = np.reshape(x_test,(len(x_test),input_rows,input_cols))
    y_test = np.reshape(y_test,(len(y_test),input_rows,input_cols))
    
    x_test = np.repeat(x_test[:, :, np.newaxis], pad_dim, axis=2)
    y_test = np.repeat(y_test[:, :, np.newaxis], pad_dim, axis=2)
    print('After reshape:')
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    print('Shuffling in unison')
    shuffle_in_unison(x_test,y_test)

    model = Unet(pretrained_weights=persistance_path)
    predictions = model.predict(x_test,verbose=0)
    print(predictions.shape)
    p_p_y = np.array([[0.0,0.0],[0.0,0.0]])
    ct = np.array([0.0,0.0])
    p_p = np.array([0.0,0.0])
    p_y = np.array([0.0,0.0])
    for j in range(0,num_of_files):
        #filename = open("results/res_test_exmpl_"+str(j)+".csv",'w')
        for i in range(0,k):
            
            #filename.write(str(predictions[j][i])+","+str(y_test[j][i])+","+str(x_test[j][i])+"\n")
            #if(y_test[j][i][0][0] != x_test[j][i][0][0]):
            #print(str(y_test[j][i][0][0])+" "+str(x_test[j][i][0][0])+"  "+str(predictions[j][i][0][0]))
            if( y_test[j][i] == 0):
                p_p_y[1][0] = p_p_y[1][0]+predictions[j][i][0][0]
            #p[1][y_test[j][i]] = p[1][y_test[j][i]]+(1.0-predictions[j][i])
                ct[0] = ct[0]+1.0
            else:
                p_p_y[1][1] = p_p_y[1][1]+predictions[j][i][0][0]
            #p[0][y_test[j][i]] = p[0][y_test[j][i]]+(1.0-predictions[j][i])
                ct[1] = ct[1]+1.0
            p_p[1] = p_p[1]+predictions[j][i][0][0]

        #filename.close()

    p_p_y[1][0] = p_p_y[1][0]/ct[0]
    p_p_y[1][1] = p_p_y[1][1]/ct[1]

    p_p_y[0][0] = 1.0 - p_p_y[1][0]
    p_p_y[0][1] = 1.0 - p_p_y[1][1]


    p_p[1] = p_p[1]/(ct[0]+ct[1])
    p_p[0] = 1-p_p[1]

    p_y[0] = ct[0]/(ct[0]+ct[1])
    p_y[1] = ct[1]/(ct[0]+ct[1])

    mut_inf = 0.0

    for i in range(0,2):
        for j in range(0,2):
            p_p_y[i][j] = p_p_y[i][j]*p_y[j]
    

    #ct = ct/sum(ct)
    #ct_pred = ct_pred/sum(ct_pred)
    file_path = "../results/"+output_file_name+".txt"
    file = open(file_path,'w+')
    file.write("Joint:\n")
    file.write(str(p_p_y))
    #print(sum(p_p_y))
    file.write("\n Marginal: Predictions:\n")
    file.write(str(p_p))
    #print(sum(p_p))
    file.write("\n Marginal: Labels:\n")
    file.write(str(p_y))

    for i in range(0,2):
        for j in range(0,2):
            if(p_p_y[i][j]!=0):
                mut_inf = mut_inf+(p_p_y[i][j]*log((p_p_y[i][j])/(p_p[i]*p_y[j]))/log(2.0))


    file.write("\nMutual information:\n")
    file.write(str(mut_inf))

if __name__ == "__main__":
    training(4095, "html","html_unet_orig","../4095_noisy_randomized_0p01/training/html_training_noisy.txt","../4095_noisy_randomized_0p01/validation/html_validation_noisy.txt")


