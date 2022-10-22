from socket import *
from struct import pack
import threading
import time

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
        self.weight = []
        for i in range(num):
            self.weight.append(1.0/num)
        self.n = num

    def aggregate(self, delta, m_weights):
        delta = np.array(delta)
        temp = self.weight[0] * np.array(m_weights[0])
        for i in range(1, self.n):
            temp += self.weight[i] * np.array(m_weights[i])
        temp -= delta
        delta += temp
        return delta


def receive(socket, i):
    # message = []
    # while True:
    #     packet = socket.recv(4096)
    #     if not packet: break
    #     message.append(packet)
    #     if len(packet) < 4096:
    #         break
    # message = b"".join(message)
    # print("receive ", i, "th client weights", len(message))
    # if len(message) > 10:
    #     weights[i] = pickle.loads(message)
    #     global count
    #     mutex.acquire()
    #     count += 1
    #     mutex.release()
    message = socket.recv(204800000)
    print("receive ", i, "th client weights", len(message))
    if len(message) > 10:
        weights[i] = pickle.loads(message)
        global count
        mutex.acquire()
        count += 1
        mutex.release()
        

weights = []
count = 0
mutex = threading.Lock()


if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('CLIENT_NUM', type=int, default=5,
                        help='client number')
    parser.add_argument('ip_address', type=str,
                        help='server ip address')
    parser.add_argument('--port', type=int, default=1200,
                        help='listen port')
    parser.add_argument('--epoch', type=int, default=25,
                        help='epoch')
    args = parser.parse_args()
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    model = get_model(7)
    model.compile(loss=loss, optimizer=opt)
    connectionSocket = []
    for _ in range(args.CLIENT_NUM):
        connectionSocket.append(0)
        weights.append(0)
    global_weights = model.get_weights()
    aggregator = Aggregator(args.CLIENT_NUM)
    # 建立连接
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind((args.ip_address, args.port))
    serverSocket.listen(args.CLIENT_NUM)
    for i in range(args.CLIENT_NUM):
        print("Waiting ", i, "th client join...")
        connectionSocket[i], address = serverSocket.accept()
        print(address, "connected!")
    for i in range(args.CLIENT_NUM):
        print("Send start signal")
        connectionSocket[i].send(b'st')
    for j in range(args.epoch):
        print("ep", j)
        count = 0
        for i in range(args.CLIENT_NUM):
            threads = threading.Thread(target=receive, args=(connectionSocket[i], i))
            threads.start()
        while True:
            if count != 0 and count % args.CLIENT_NUM == 0:
                break
        global_weights = aggregator.aggregate(global_weights, np.array(weights))
        for i in range(args.CLIENT_NUM):
            print("Sending new weights")
            connectionSocket[i].sendall(pickle.dumps(global_weights))
    for i in range(args.CLIENT_NUM):
        connectionSocket[i].close()
    # Save final result
    model.set_weights(global_weights)
    model.save("./final_model.hdf5")
