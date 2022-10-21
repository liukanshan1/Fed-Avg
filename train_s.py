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
        # self.weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.weight = np.array([1])

    def aggregate(self, delta, m_weights):
        delta = np.array(delta)
        temp = np.dot(self.weight, m_weights)
        temp -= delta
        delta += temp
        return delta


if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('ip_address', type=str,
                        help='server ip address')
    parser.add_argument('port', type=int, default=1200,
                        help='listen port')
    parser.add_argument('CLIENT_NUM', type=int, default=5,
                        help='client number')
    args = parser.parse_args()
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_data, args.dataset_name, args.path_to_annotations, batch_size, args.val_split)
    model = get_model(train_seq.n_classes)
    model.compile(loss=loss, optimizer=opt)
    # 建立连接
    print("Waiting...")
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind((args.ip_address, args.port))
    serverSocket.listen(args.CLIENT_NUM)
    connectionSocket, address = [], []
    weights = []
    for _ in range(args.CLIENT_NUM):
        connectionSocket.append(0)
        address.append(0)
        weights.append(0)
    global_weights = model.get_weights()
    aggregator = Aggregator()
    for i in range(args.CLIENT_NUM):
        connectionSocket[i], address[i] = serverSocket.accept()
        print(address, "connected!")
        print("Send start signal")
        connectionSocket[i].send(b'st')
    for j in range(25):
        print("ep", j)
        for i in range(args.CLIENT_NUM):
            message = connectionSocket[i].recv(204800000)
            print("receive ", i, "th client weights")
            if len(message) > 10:
                weights[i] = pickle.loads(message)
        global_weights = aggregator.aggregate(global_weights, np.array(weights))
        for i in range(args.CLIENT_NUM):
            print("Sending new weights")
            connectionSocket[i].send(pickle.dumps(global_weights))
    for i in range(args.CLIENT_NUM):
        connectionSocket[i].close()
    # Save final result
    model.set_weights(global_weights)
    model.save("./final_model.hdf5")
