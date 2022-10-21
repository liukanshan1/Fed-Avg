from socket import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from model import get_model
import argparse
from datasets import ECGSequence
import numpy as np
import pickle


class Aggregator:

    def __init__(self):
        self.weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def aggregate(self, delta, m_weights):
        delta = np.array(delta)
        temp = np.dot(self.weight, m_weights)
        temp -= delta
        delta += temp
        return delta


if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_to_data', type=str,
                        help='path to data dir containing tracings')
    parser.add_argument('path_to_annotations', type=str,
                        help='path to dir containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='val',
                        help='name of the dataset containing tracings')
    args = parser.parse_args()
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_data, args.dataset_name, args.path_to_annotations, batch_size, args.val_split)

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(train_seq.n_classes)
    model.compile(loss=loss, optimizer=opt)
    # Create log
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.hdf5'),
                  ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]
    
    # 建立连接
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind(("192.168.1.25"), 1200)
    serverSocket.listen(6)
    connectionSocket, address = [], []
    weights = np.array([])
    global_weights = model.get_weights()
    aggregator = Aggregator()
    for i in range(5):
        connectionSocket[i], address[i] = serverSocket.accept()
        connectionSocket[i].send(b'st')
    for _ in range(25):
        for i in range(5):
            message = connectionSocket[i].recv(204800)
            if len(message) > 10:
                weights[i] = pickle.loads(message)
        global_weights = aggregator.aggregate(global_weights, weights)
        for i in range(5):
            connectionSocket[i].send(pickle.dump(global_weights))
    for i in range(5):
        connectionSocket[i].close()
    # Save final result
    model.set_weights(global_weights)
    model.save("./final_model.hdf5")
