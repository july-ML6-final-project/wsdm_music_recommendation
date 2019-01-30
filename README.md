# wsdm_music_recommendation
Final project for July Recommendation course

## 项目概述

该问题是典型的推荐算法领域的CTR/CVR预估问题，适用于learning to rank框架。需要用到的features包括用户特征，物品特征和用户用品关系的上下文特征。
既可以采用传统的GBT模型，也可以采用DNN模型。


## 操作细节


### 硬件要求
所有的代码都可以在多核机器上进行并行操作，但是注意有些代码生成特征比较耗时间，如svd。建议使用120GB以上的内存以及GPU（跑深度网络代码）
此外，需要10-20GB的磁盘空间来存储各种生成的特征文件。


### 代码运行步骤
1. **预处理**.
    运行`codes/feature/id_process.py`将所有的categorical特征进行label encoding

2. **特征生成**.

    1）进入`codes/feature`
    2）运行`cnt_log_process.py`生成统计类特征
    3）运行`isrc_process.py`处理歌曲特殊代码
    4）运行`svd_process.py`矩阵分解得到embedding特征
    5）运行`timestamp_process.py`生成时序类特征
    6) 运行`before_after_process.py`生成user-song对的上下文统计特征
    7）运行`data_for_training.py`和`experiments.py`将所有生成的特征合并成可以用于训练的数据
    
    
    *注意*: 建议在有GPU和NVIDIA CUDA的机子上运行所有的`nn_*.py`文件。
    
3. **模型训练**.

	1）进入`codes/training`
	2) 运行`lgb_training.py`训练LGB模型并产生预测结果数据
	3）运行`nn_training.py`训练多个DNN模型并产生预测数据
	4）运行`nn_ensemble.py`对DNN预测结果进行融合
	5）运行`lgb_nn_ensemble.py`对LGB+DNN的结果进行融合并产生最终提交数据文件

