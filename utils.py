import scipy.io
import glob

def get_all_mat(path, dset):
    mat_files = get_all_files(path, '.mat')
    mats = []
    for mat_file in mat_files:
        mats.append(scipy.io.loadmat(mat_file, None)[dset])
    return mats


def get_all_hea(path):
    return get_all_files(path, '.hea')


def get_all_files(path, file_type):
    files = []
    for file_abs in glob.glob(path): #TODO 绝对路径，优化
        if file_abs.endswith(file_type):
            files.append(file_abs)
    return files


