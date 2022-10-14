import scipy.io
import os


def get_all_mat(path, dset):
    mat_files = get_all_files(path, '.mat')
    mats = []
    for mat_file in mat_files[0:5]:
        mats.append(scipy.io.loadmat(mat_file, None)[dset])
    return mats


def get_all_hea(path):
    hea_files = get_all_files(path, '.hea')
    heas = []
    for hea_file in hea_files:
        with open(hea_file, 'r') as f:
            line = f.readlines()[15]  # 读取第15行
            dxs = line[5:-1].split(',')
            for dx in dxs:
                if dx == '164889003':
                    heas.append([1, 0, 0, 0, 0, 0, 0])
                    break
                elif dx == '164890007':
                    heas.append([0, 1, 0, 0, 0, 0, 0])
                    break
                elif dx == '713422000':
                    heas.append([0, 0, 1, 0, 0, 0, 0])
                    break
                elif dx == '426177001':
                    heas.append([0, 0, 0, 1, 0, 0, 0])
                    break
                elif dx == '426783006':
                    heas.append([0, 0, 0, 0, 1, 0, 0])
                    break
                elif dx == '427084000':
                    heas.append([0, 0, 0, 0, 0, 1, 0])
                    break
                elif dx == '426761007':
                    heas.append([0, 0, 0, 0, 0, 0, 1])
                    break
    return heas


def get_all_files(path, file_type):
    files = []
    for file_name in os.listdir(path):
        if file_name.endswith(file_type):
            files.append(path + file_name)
    return files


def get_file_num(path, file_type):
    num = 0
    for file_name in os.listdir(path):
        if file_name.endswith(file_type):
            num += 1
    return num


if __name__ == '__main__':
    path = './newData/'
    dset = 'val'
    a = get_all_mat(path, dset)
    print(type(a))
    print(a)
    print(type(a[0]))
    print(a[0])
    print(type(a[0][0]))
    print(a[0][0])