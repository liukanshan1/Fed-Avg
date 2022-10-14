from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

# 回调——————打印总训练时间（从最初到当前epoch所花费的时间）
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = tf.timestamp()
    def on_epoch_end(self,epoch,logs = {}):
        # self.times.append(tf.timestamp() - self.timetaken)
        t=tf.timestamp() - self.timetaken
        self.times.append(t)
        print('cost time: ',t.numpy())
        self.epochs.append(epoch)
    def on_train_end(self,logs = {}):
        # plt.xlabel('Epoch')
        # plt.ylabel('Total time taken until an epoch in seconds')
        # plt.plot(self.epochs, self.times, 'ro')
        for i in range(len(self.epochs)):
          j = self.times[i].numpy()
          if i == 0:
            plt.text(i, j, str(round(j, 3)))
          else:
            j_prev = self.times[i-1].numpy()
            plt.text(i, j, str(round(j-j_prev, 3)))
        # plt.savefig(datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
