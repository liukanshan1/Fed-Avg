import scipy.io

def get_all_mat(path, dset):
    mat_files = get_all_files(path, '.mat')
    mats = []
    for mat_file in mat_files:
        mats.append(scipy.io.loadmat(mat_file, None)[dset])
    return mats


def get_all_hea(path):
    return get_all_files(path, '.hea')


def get_all_files(path, file_type):
    root, dirs, files = os.walk(path)
    result = []
    for file in files:
        if file.endswith(file_type):
            result.append(file_abs)
    return result

if __name__ == '__main__':
    path = './newData/'
    dset = 'val'
    #print(get_all_mat(path, dset))
    print(get_all_files(path, ".mat"))
