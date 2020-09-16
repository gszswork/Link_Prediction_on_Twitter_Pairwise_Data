# Link-Prediction-on-Possible-Twitter-Pairwise-Relationship
COMP90051 Statistical Machine Learning Project 1


## Report Position
https://enf8hwgu3f.feishu.cn/docs/doccnVaAKn0TjkYziR28kLM5Tuf


## How to run
The files are all in jupyter. 

#### 1. Feature_Engineering.ipynb: 
Use train.txt to generate pos_data and neg_data and extract the primary features (pre_features).

input files:
```
train.txt
```
output files:
```
pre_features_v2.pkl
SB_data.pkl
```
Notes: SB_data is a funny nick name named by them not me lol. it contains pos and neg edges in format [(source, sink) , label].

#### 2_Gen_training_data.ipynb:
Use the test-public.txt to generate edges list in training data related to the nodes in test-public.txt file.

input files:
```
SB_data.pkl
test-public.txt
```

output file:
```
edges_of_all_test_nodes_related.pkl
```

#### 3_Gen_final_features.ipynb:
Use the 'edge_of_all_test_nodes_related' and 'pre_features' to generate the final training features named 'training_df'.
The training_df is in DataFrame format and is very large to generate. To speed up we split and compute in parrallel.

input files:
```
pre_features_v2.pkl
edges_of_all_test_nodes_related.pkl
```

output files:
```
training_df.pkl
```
