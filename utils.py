import scipy.io
import os

def get_all_mat(path, dset):
    mat_files = get_all_files(path, '.mat')
    mats = []
    for mat_file in mat_files:
        mats.append(scipy.io.loadmat(mat_file, None)[dset])
    return mats


def get_all_hea(path):
    hea_files = get_all_files(path, '.hea')
    for hea_file in hea_files:
        with open(hea_file, 'r') as f:
            line = f.readlines()[15] # 读取第15行
            dxs = line[5:-1].split(',')



def get_all_files(path, file_type):
    files = []
    for file_name in os.listdir(path):
        if file_name.endswith(file_type):
            files.append(path + file_name)
    return files

if __name__ == '__main__':
    path = './newData/'
    dset = 'val'
    get_all_hea(path)
    #print(get_all_files(path, ".mat")[0:10])
