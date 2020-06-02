# DCIC 2020数字中国创新大赛数字政府赛道：智慧海洋建设Rank 3解决方案

---
### 队伍简介
liu123的航空母舰队，队长鱼丸粗面(zhuoyin94@163.com)。复赛算法阶段F1成绩0.8995(3/3275)，复赛可视化阶段成绩21.0(7/14)。注：以上Rank为算法赛阶段成绩。


### 主要依赖packages与运行依赖环境
- 系统环境: Ubuntu 18.04 LTS
- python: 3.7.1
- gensim: 3.8.1
- sklearn: 0.22.1
- pandas: 1.0.3
- lightgbm: 2.3.1
- xgboost 1.0.1
- geopandas: 0.7.0

---
### 基本思路说明
本项目采用了传统统计机器学习建模与轨迹数据挖掘[1]的思路。特征工程主要包括两部分：基础统计特征与轨迹embedding特征；模型方面采用了XGBoost和LightGBM作为基模型。以下为简单介绍：

- **预处理**: 轨迹数据预处理方面, 首先采用了经验阈值滤除了每条轨迹速度的异常值、坐标的离群点， 并用多项式插值函数对离群点进行了插值。我们也探索了许多的平滑方法用于滤除噪声坐标，例如Savitzky–Golay Filter[2], 中值滤波，Kalman Filter[3]等， 总的来说这些方法不如基于阈值的均值滤波来的简单有效。更有效的清洗方法参见文献[7]。

- **POI信息挖掘**： 我们采用了基于经验的POI挖掘策略。具体来说，我们将每条轨迹投射到网格坐标系下，这样轨迹序列变为了网格id的符号序列； 随后我们基于被boat_id不同的渔船访问次数， 不同boat_id的渔船在该网格停留的平均时长和网格总的被访问的次数三个判据，筛选出了一系列的POI网格。我们同样尝试了一些无监督的轨迹语义挖掘方法，例如文献[6]以及基于两阶段聚类的ROI区域聚类， 总的来说效果和效率都不容易控制。

- **特征工程**: 特征工程分为两部分, 第一部分为基础统计特征, 对于每条轨迹的x与y坐标, 速度与方向以及一些交叉的结果提取了分位数, 方向直方图, 地理位置信息等基础统计信息; 第二部分为word embedding的特征. 我们将每条轨迹的坐标所在的网格id视为一个词, 每条轨迹视为一个句子。随后对每一个词做了word embedding[4] [5], 每条句子的句子向量为句子包含词的向量的平均, 可直接作为特征feed进统计模型。

- **机器学习算法**: 直接采用了LightGBM和XGBoost作为基模型， 第二层Stacking使用LightGBM作为Stacking模型。事实上由于测试集大小问题，第二层模型采用直接平均法线下效果更佳， 囿于评测机会不多没有做线上测评。

- **随机性处理**: 由于gensim采用了多线程加快训练速度， 由于OS在调配资源时会有些许不同，这就导致w2v的下采样得到的词会有些许不同，进而会导致相同参数训练的w2v的词向量不一致[8]。在比赛中这个随机性对最终结果影响较大（F1线下大概在0.914到0.918之间振荡）。因此我们训练了多组的embedding, 尽量缓解随机性带来的影响。（虽然这个解法并不优雅，实际中也没法说明效果，有更好的解法请在issue中指出）

- **半监督学习**: 我们采用了train + test_a + test_b的数据集进行无监督的word embedding, 采用train预测出来的test_a的标签作为伪标签填充train集做数据增强。（思路很简单，源码没有包含。）

- **AIS轨迹与北斗轨迹的匹配**: 后期主办方提供了AIS轨迹数据。AIS轨迹可以被认为是渔船不同来源的轨迹，统一艘渔船可能既有北斗轨迹也有AIS轨迹，但是二者之间的关系需要自行匹配。我们设计了一个两阶段的匹配策略，具体细节参见答辩PPT。（源码未包含）

---
### 代码文件说明

#### 预处理部分
- **traj_data_train_test_split.py**: 基于比赛数据，分层采样出训练集与测试集数据，方便线下调试，以适应线上评测的Docker环境。
- **traj_data_preprocessing.py**: 预处理每一条轨迹数据，完成以下工作：基于局部速度对异常坐标点进行插值；基于经验对异常速度进行插值；将WGS-84(EPSG:4326)转为EPSG:3395 Mercator坐标，方便坐标距离的计算。
- **ais_data_preprocessing.py**: 对AIS轨迹数据执行以上相似操作。

#### POI信息挖掘部分
- **traj_data_poi_mining.py**: 基于规则挖掘POI信息。
- **traj_data_labeling_semantics.py**: 依据所挖掘的POI信息，为每一条训练样本和测试样本分配POI标签。

#### Embedding部分
- **embedding_geo_information.py**: 用于对坐标信息进行embedding。我们测试了Skip-Gram和CBOW两种模型，最后仅使用了CBOW作为我们的模型。
- **embedding_signal_sequence.py**: 用于针对渔船的速度与方向序列进行embedding。

#### 特征工程部分
- **stat_feature_engineering_lgb.py**: 针对LightGBM模型的特征工程。
- **stat_feature_engineering_xgb.py**: 针对XGBoost模型的特征工程。

#### 模型训练
- **traj_data_classification.py**： 最终的分类器训练。

---
### References
[1] Zheng Y . Trajectory Data Mining: An Overview[J]. ACM Transactions on Intelligent Systems and Technology, 2015, 6(3):1-41.

[2] Schafer R W. What Is a Savitzky-Golay Filter? [Lecture Notes][J]. IEEE Signal Processing Magazine, 2011, 28(4): 111-117.

[3] interesting Kalman filter links. An Introduction to the Kalman Filter[J]. 1995.

[4] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. ICLR Workshop, 2013

[5] T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean. Distributed Representations of Words and Phrases and their Compositionality. NIPS 2013

[6] Palma A T, Bogorny V, Kuijpers B, et al. A clustering-based approach for discovering interesting places in trajectories[C]//Proceedings of the 2008 ACM symposium on Applied computing. 2008: 863-868.

[7] Zhang A, Song S, Wang J. Sequential data cleaning: a statistical approach[C]//Proceedings of the 2016 International Conference on Management of Data. 2016: 909-924.

[8] https://github.com/RaRe-Technologies/gensim/issues/641