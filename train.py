from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
import tensorflow as tf
from model import get_model
import argparse
from datasets import ECGSequence

def focal_loss(logits, labels, epsilon=1.e-7,
               gamma=2.0,
               multi_dim=False):
    '''
        :param logits:  [batch_size, n_class]
        :param labels: [batch_size]  not one-hot !!!
        :return: -alpha*(1-y)^r * log(y)
        它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
        logits soft max之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

        怎么把alpha的权重加上去？
        通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

        是否需要对logits转换后的概率值进行限制？
        需要的，避免极端情况的影响

        针对输入是 (N，P，C )和  (N，P)怎么处理？
        先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

        bug:
        ValueError: Cannot convert an unknown Dimension to a Tensor: ?
        因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

        '''

    # 注意，alpha是一个和你的分类类别数量相等的向量；
    alpha = [[1], [1], [1], [1], [1], [1], [1]]

    if multi_dim:
        logits = tf.reshape(logits, [-1, logits.shape[2]])
        labels = tf.reshape(labels, [-1])

    # (Class ,1)
    alpha = tf.constant(alpha, dtype=tf.float32)

    labels = tf.cast(labels, dtype=tf.int32)
    logits = tf.cast(logits, tf.float32)
    # (N,Class) > N*Class
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
    # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
    labels_shift = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
    # labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
    # (N*Class,) > (N,)
    prob = tf.gather(softmax, labels_shift)
    # 预防预测概率值为0的情况  ; (N,)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    # (Class ,1) > (N,)
    alpha_choice = tf.gather(alpha, labels)
    # (N,) > (N,)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    # (N,) > 1
    loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
    return loss


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
    # Optimization settings
    loss = focal_loss
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
    # Train neural network
    history = model.fit(train_seq,
                        epochs=70,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)
    # Save final result
    model.save("./final_model.hdf5")
