import scipy.io

a = scipy.io.loadmat('./newData/JS00002.mat', 'val')
print(a)

import glob

def get_all_mat(path):



def get_all_files(path, file_type):
    for file_abs in glob.glob(path) :
    print(file_abs)


