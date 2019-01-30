##############################################################################################
########################### LightGBM + NN ensemble/blending ##################################
##############################################################################################

import pandas as pd

lgb_weight = 0.6
nn_weight = 0.4

lgb_results = "./submission/lgb_0.73963_740.csv.gz"
nn_results = "./submission/85.48780_41_ensemble_add.csv.gz"

lgb_pred = pd.read_csv(lgb_results)
nn_pred = pd.read_csv(nn_results)

blending_pred = pd.concat([lgb_pred['id'], lgb_weight * lgb_pred["target"] + nn_weight * nn_pred["target"]], \
				axis=1)
blending_pred.to_csv("./submission/lgb_nn_blending3.csv.gz", index=False, compression='gzip')
