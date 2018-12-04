# coding: utf-8
# An attempt to use UNet on the RACC_NR problem.
# (Command line usage available)

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
from keras.callbacks import CSVLogger, EarlyStopping
from keras import losses
from keras import regularizers
from models import Unet1D, DilatedUnet1D, Unet1DNoPooling
import os
import argparse

# Initial setup
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
K.set_session(tf.Session(config = config))

def shuffle_in_unison(a, b):
    """ 
        Shuffling the data. Order does not matter.
        
    Parameters: 
        a (np.ndarray): the first item to be shuffled.
        b (np.ndarray): the second item to be shuffled.
        
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def get_next_character(f):
    """ 
        Reads one character from the given textfile.
        
    Parameters: 
        f (file): the file object to be read.
        
    """
    c = f.read(1)
    while c:
        yield c
        c = f.read(1)

def load_data(ground_truth_folder, noisy_folder, label):
    """ 
        Preparing the data.
        
    Parameters: 
        ground_truth_folder (str): the path to the ground truth folder.
        noisy_folder (str): the path to the noisy folder.
        label (str): type of the data. Could be one of the following: pdf, latex, jpeg, html.
        
    Returns:
        np.array(x): an numpy array representation of the features (X).
        np.array(y): an numpy array representation of the labels (Y).
        filenames (list of str): a collection of files inspected.
    """
    x = []
    y = []
    filenames = []
    count = 0
    
    if ground_truth_folder[-1] != '/':
        ground_truth_folder += '/'
    if noisy_folder[-1] != '/':
        noisy_folder += '/'
    
    all_files = os.listdir('%s' % ground_truth_folder)
    
    for file in all_files:
        count += 1
        if count % 500 == 0:
            print("Loading "+kind+" example: "+str(count)+" for label : "+label)
        
        filenames.append(file)
            
        with open(ground_truth_folder+label+"/"+file, 'rb') as f:
            l = []
            for c in get_next_character(f):
                try:
                    l.append(int(c))
                except ValueError:
                    print("Error in ground truths:", c, file)
            l = np.array(l)
            y.append(l) 
        
        with open(noisy_folder+label+"/"+file, 'rb') as f:
            l = []
            for c in get_next_character(f):
                try:
                    l.append(int(c))
                except ValueError:
                    print("Error in noises:", c, file)
            l = np.array(l)
            x.append(l)
            
    return (np.array(x), np.array(y),filenames)

def negGrowthRateLoss(b,q):
    """ Customized loss function. """
    return (K.mean(-K.log(b +pow(-1,b)+pow(-1,b+1)*q)/K.log(2.0)))

def training(k, fileType, train_ground_truth_folder, train_noisy_folder, val_ground_truth_folder, val_noisy_folder, dest):
    """ 
        Training the data.
        
    Parameters: 
        k (int): number of bits to look at.
        fileType (str): type of the file. Could be one of the following: pdf, latex, jpeg, html.
        train_ground_truth_folder (str): the path to the training ground truth folder.
        train_noisy_folder (str): the path to the training noisy folder.
        val_ground_truth_folder (str): the path to the validation ground truth folder.
        val_noisy_folder (str): the path to the validation noisy folder.
        dest (str): the path to the destination folder. For training, the model presistence will be saved.
    """
    input_rows, input_cols = k, 1
    #pad_dim = 32
    
    # the data, shuffled and split between train and test sets
    (x_train, y_train,_) = load_data(train_ground_truth_folder, train_noisy_folder, fileType)
    (x_valid, y_valid,_) = load_data(val_ground_truth_folder, val_noisy_folder, fileType)
    print('Before reshape:')
    print('x_train shape:', x_train.shape)
    print('x_valid shape:', x_valid.shape)
    # add a zero column at the end
    x_train = np.append(x_train,np.zeros([len(x_train),1]),1)
    x_valid = np.append(x_valid,np.zeros([len(x_valid),1]),1)
    y_train = np.append(y_train,np.zeros([len(y_train),1]),1)
    y_valid = np.append(y_valid,np.zeros([len(y_valid),1]),1)
    
    # reshaping
    x_train = np.reshape(x_train,(len(x_train),input_rows,input_cols,1))
    x_valid = np.reshape(x_valid,(len(x_valid),input_rows,input_cols,1))
    y_train = np.reshape(y_train,(len(y_train),input_rows,input_cols,1))
    y_valid = np.reshape(y_valid,(len(y_valid),input_rows,input_cols,1))

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

    if dest[-1] != '/':
        dest += '/'
    if dest.split('/')[-2].lower() != fileType:
        dest += fileType + '/'
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    # below is an example for the html U-Net model
    model = Unet1DNoPooling(input_size=input_shape, k = k, loss = negGrowthRateLoss, learning_rate = 1e-5)
    filepath = dest+fileType+".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(dest+fileType+".csv")
    # Helper: Early stopping.
    early_stopper = EarlyStopping(patience=5)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid),callbacks=[checkpoint,csv_logger,early_stopper])
    model.save_weights(filepath)

def inference(k, persistance_path, fileType, test_ground_truth_folder, test_noisy_folder, dest):
    """ 
        Using the model presistence to do predictions.
        
    Parameters: 
        k (int): number of bits to look at.
        fileType (str): type of the file. Could be one of the following: pdf, latex, jpeg, html.
        persistance_path (str): the path to the saved model file.
        test_ground_truth_folder (str): the path to the testing ground truth folder.
        test_noisy_folder (str): the path to the testing noisy folder.
        dest (str): the path to the destination folder. For inference, the actual perdiction and the summary file will be saved.
    """
    input_rows, input_cols = k, 1
    #pad_dim = 32
    
    # load the shuffled test data
    (x_test, y_test,filenames) = load_data(test_ground_truth_folder, test_noisy_folder, fileType)
    print('Before reshape:')
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    # add a zero column at the end
    x_test = np.append(x_test,np.zeros([len(x_test),1]),1)
    y_test = np.append(y_test,np.zeros([len(y_test),1]),1)
    x_test = np.reshape(x_test,(len(x_test),input_rows,input_cols,1))
    y_test = np.reshape(y_test,(len(y_test),input_rows,input_cols,1))
    
    print('After reshape:')
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    print('Shuffling in unison')
    shuffle_in_unison(x_test,y_test)
    
    input_shape = (input_rows, input_cols, 1)

    model = Unet1DNoPooling(input_size=input_shape, k = k, loss = negGrowthRateLoss, pretrained_weights=persistance_path, learning_rate = 1e-5)
    predictions = model.predict(x_test,verbose=0)
    print(predictions.shape)
    
    p_p_y = np.array([[0.0,0.0],[0.0,0.0]])
    ct = np.array([0.0,0.0])
    p_p = np.array([0.0,0.0])
    p_y = np.array([0.0,0.0])
    
    if dest[-1] != '/':
        dest += '/'
    if dest.split('/')[-2].lower() != fileType:
        dest += fileType + '/'
    if not os.path.exists(dest):
        os.makedirs(dest)
        
        
    for j in range(0,len(y_test)):
        filename = open(dest+filenames[j],'w+')
        # Excluding the last bit here, since it's appended and will always be 0
        for i in range(0,4095):
            
            filename.write(str(predictions[j][i])+"\n")
            #if(y_test[j][i][0][0] != x_test[j][i][0][0]):
            #print(str(y_test[j][i][0][0])+" "+str(x_test[j][i][0][0])+"  "+str(predictions[j][i][0][0]))
            if(y_test[j][i] == 0):
                p_p_y[1][0] = p_p_y[1][0]+predictions[j][i][0][0]
            #p[1][y_test[j][i]] = p[1][y_test[j][i]]+(1.0-predictions[j][i])
                ct[0] = ct[0]+1.0
            else:
                p_p_y[1][1] = p_p_y[1][1]+predictions[j][i][0][0]
            #p[0][y_test[j][i]] = p[0][y_test[j][i]]+(1.0-predictions[j][i])
                ct[1] = ct[1]+1.0
            p_p[1] = p_p[1]+predictions[j][i][0][0]

        filename.close()

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
    file_path = dest+fileType+"test_summary.txt"
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
    # main program
    parser = argparse.ArgumentParser(
        description='RACC_NR U-Net script')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('-trgt','--traingt', required=False,
                        metavar="/path/to/dataset/",
                        help='Ground truth folder for the training data')
    parser.add_argument('-trn','--trainn', required=False,
                        metavar="/path/to/dataset/",
                        help='Noisy folder for the training data')
    parser.add_argument('-vgt','--valgt', required=False,
                        metavar="/path/to/dataset/",
                        help='Ground truth folder for the validation data')
    parser.add_argument('-vn','--valn', required=False,
                        metavar="/path/to/dataset/",
                        help='Noisy folder for the validation data')
    parser.add_argument('-tgt','--testgt', required=False,
                        metavar="/path/to/dataset/",
                        help='Ground truth folder for the testing data')
    parser.add_argument('-tn','--testn', required=False,
                        metavar="/path/to/dataset/",
                        help='Noisy folder for the testing data')
    parser.add_argument("--ft", required=True,
                        metavar="html",
                        help="Available file type: html, latex, jpeg, pdf")
    parser.add_argument('-d','--dest', required=False,
                        metavar="/path/to/prediction/",
                        help='Path to save the prediction files (at the prediction time) or model file (at the training time)')
    parser.add_argument('-p','--persistence', required=False,
                        metavar="/path/to/persistence/",
                        help='Path to the saved model file')
    args = parser.parse_args()
    
    # validation
    if args.command == "train":
        assert args.traingt, "Argument --traingt is required for training"
        assert args.trainn, "Argument --trainn is required for training"
        assert args.valgt, "Argument --valgt is required for training"
        assert args.valn, "Argument --valn is required for training"
    elif args.command == "test":
        assert args.testgt, "Provide --testgt to run prediction on"
        assert args.testn, "Provide --testn to run prediction on"
        assert args.persistence, "Provide --persistence to load the saved model"
        
    # Actual running
    if args.command == "train":
        training(4096, args.ft, args.traingt, args.trainn, args.valgt, args.valn, args.dest)
    elif args.command == "test":
        inference(4096, args.persistence, args.ft, args.testgt, args.testn, args.dest)

