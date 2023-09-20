# 联邦学习ECG心电图诊断模型训练与预测
采用明文模型平均，模型结构和部分参考论文Automatic diagnosis of the 12-lead ECG using a deep neural network及其开源实现
 
 ## Scripts

- ``train.py``: Script for training the neural network. To train the neural network run: 
```bash
$ python train.py .\data\ .\data\
```

- ``predict.py``: Script for generating the neural network predictions on a given dataset.
```bash
$ python predict.py .\data\test_set\ .\final_model.hdf5
```

- ``generate_figures_and_tables.py``: Generate results.
 ```bash
$ python generate_results.py
 ```
