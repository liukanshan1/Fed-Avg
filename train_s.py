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

    def __init__(self, num):
        # self.weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        weight = []
        for i in range(num):
            weight.append(1.0/num)
        self.weight = np.array(weight)

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
    model = get_model(7)
    model.compile(loss=loss, optimizer=opt)
    connectionSocket, address = [], []
    weights = []
    for _ in range(args.CLIENT_NUM):
        connectionSocket.append(0)
        address.append(0)
        weights.append(0)
    global_weights = model.get_weights()
    aggregator = Aggregator(args.CLIENT_NUM)
    # 建立连接
    for i in range(args.CLIENT_NUM):
        print("Waiting ", i, "th client join...")
        serverSocket = socket(AF_INET, SOCK_STREAM)
        serverSocket.bind((args.ip_address, args.port + i))
        serverSocket.listen(1)
        connectionSocket[i], address[i] = serverSocket.accept()
        print(address, "connected!")
    for i in range(args.CLIENT_NUM):
        print("Send start signal")
        connectionSocket[i].send(b'st')
    for j in range(25):
        print("ep", j)
        for i in range(args.CLIENT_NUM):
            message = connectionSocket[i].recv(20480000000)
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
