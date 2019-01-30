#########################################################################################
################# Take the mean of all NN model output as ensemble output################
#########################################################################################


import os

import pandas as pd

cnt = 0.0
score = 0.0

for item in os.listdir('../../data/submissions/temp_nn/'):
    score += float(item.split('_')[1])
    tmp = pd.read_csv('../../data/submissions/temp_nn/'+item)
    if cnt == 0:
        preds = tmp
    else:
        preds['target'] += tmp['target']
    cnt += 1.0

score /= cnt
preds['target'] /= cnt
preds.to_csv('../../data/submissions/%.5f_%d_ensemble_add.csv.gz'%(score, cnt), index=False, \
        compression='gzip')

