from socket import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
import tensorflow as tf
from model import get_model
import argparse
from datasets import ECGSequence
import pickle


def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
    当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    # alpha = tf.constant([[0.96098], [0.80820], [0.99651], [0.61015], [0.80857], [0.83103], [0.98454]], dtype=tf.float32)
    # alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)

    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss1_fixed

def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss2_fixed


if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_to_data', type=str,
                        help='path to data dir containing tracings')
    parser.add_argument('path_to_annotations', type=str,
                        help='path to dir containing annotations')
    parser.add_argument('ip_address', type=str,
                        help='server ip address')
    parser.add_argument('port', type=int, default=1200,
                        help='listen port')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='val',
                        help='name of the dataset containing tracings')
    args = parser.parse_args()
    # Optimization settings
    # loss = [
    #     multi_category_focal_loss1(alpha=[[0.96098], [0.80820], [0.99651], [0.61015], [0.80857], [0.83103], [0.98454]],
    #                                gamma=2)]  #  'binary_crossentropy'
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,  # TODO last
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau  # TODO last
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
    print("conneting...")
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((args.ip_address, args.port))
    print("connected!")

    for i in range(25):
        print("ep", i)
        message = clientSocket.recv(204800000)
        if len(message) > 10:
            global_weights = pickle.loads(message)
            model.set_weights(global_weights)  # 从服务器获取参数
            print("get weights from server.")
        # Train neural network
        history = model.fit(train_seq,
                            epochs=2,
                            initial_epoch=0,
                            callbacks=callbacks,
                            validation_data=valid_seq,
                            verbose=1)
        weights = pickle.dumps(model.get_weights())
        print("send weights to server.")
        clientSocket.send(weights)

    clientSocket.send(b'break')
    clientSocket.close()

