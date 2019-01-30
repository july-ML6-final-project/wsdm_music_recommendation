#################################################################################################
################### generate 50 NN models with randomly selected parameters #####################
#################################################################################################

import numpy as np
import pandas as pd

nn_record = pd.read_csv("../../data/nn_record.csv")
N = 50 - len(nn_record)

np.random.seed(2019)

new_record = []
for n in range(N):
    K = np.random.randint(48, 128)  # 64
    K0 = np.random.randint(4, 16)  # 8
    lw = 7.5e-4 * (0.1 ** (np.random.rand() * 3 - 1.5))
    lw1 = 0.0  # 1e-3 * (0.1 ** (np.random.rand() * 3 - 1.5))
    lr = 7.5e-3 * (0.1 ** (np.random.rand() * 2 - 1.0))
    lr_decay = 0.65 + np.random.rand() * 0.3
    activation = np.random.choice(['relu', 'tanh', 'prelu', 'leakyrelu', 'elu'])
    batchnorm = np.random.choice([True, False])
    sample_weight_rate = 0.0
    bst_epoch = np.random.randint(20, 40)

    new_record.append({"K":K, "K0":K0, "lw":lw, "lw1":lw1, "lr":lr, "lr_decay":lr_decay, "bst_epoch":bst_epoch, \
		      "activation":activation, "batchnorm":batchnorm, "sample_weight_rate":sample_weight_rate})

  
new_record = pd.DataFrame(new_record)
new_nn_record = pd.concat([nn_record, new_record])
new_nn_record.fillna(method='ffill', inplace=True)
new_nn_record.to_csv("../../data/new_nn_record.csv", index=False)
    
    

