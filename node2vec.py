import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pickle
import math
import pandas as pd
import pickle
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

obj_data_dir = "../../data/"
def save_obj(obj, name ):
    with open(obj_data_dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(obj_data_dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)
# load data
# BasicFeatures = load_obj("pre_features")
training_edges=load_obj('SBdata')


data_dir = "../../node2vec/emb/"
cnt=0
flag=False
emb_dict = {}
with open(data_dir + "sb_data_edges.emd", "r") as f:
#for read_rows in training_egdes:
    read_rows = f.readlines()
    for row in read_rows:
        if cnt ==0:
            cnt+=1
            continue
        row_array = np.array(row.split())
        row_vector= [float(x) for x in row_array[1:]]
        emb_dict[int(row_array[0])]=row_vector

from sklearn.metrics.pairwise import cosine_similarity

# print(cosine_similarity([1, 0, -1], [-1,-1, 0]))
# array([[-0.5]])

training_edges_cosine = []
#  对training_data中的做cosine
mean =0
cnt=0
summation=0
for training_edge in training_edges:
    label = int(training_edge[1])
    source = int(training_edge[0][0])
    sink = int(training_edge[0][1])
    try:
        source_vector = np.array(emb_dict[source]).reshape(1, -1)
        sink_vector = np.array(emb_dict[sink]).reshape(1, -1)
        cosine_similarity_result = cosine_similarity(source_vector,sink_vector)[0]
        cnt +=1
        summation = cosine_similarity_result + summation
    except KeyError:
        cosine_similarity_result=None
    training_edges_cosine.append((source, sink, label, cosine_similarity_result))
mean = summation/cnt

final_training_edges_cosine=[]
for edge in training_edges_cosine:
    if edge[3] is None:
        cosine_similarity_result=mean[0]
    else:
        cosine_similarity_result=edge[3][0]

    final_training_edges_cosine.append((edge[0], edge[1], edge[2], cosine_similarity_result))

final_training_data = pd.DataFrame(final_training_edges_cosine)
final_labels_df = final_training_data[2]
final_training_data_df = final_training_data.drop(columns=2)
# >>> df.drop(columns=['B', 'C'])

# 用logistic regression试试
X=final_training_data_df
# count=0
# get the data and label
y=final_labels_df

# training model
from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test = train_test_split(X,y)
X_train, X_validation, y_train, y_validation  = train_test_split(X_t,y_t)
# Gridsearch settings
pipeline = Pipeline([
    ('clf', LogisticRegression())
])
parameters = {
       'clf__penalty': ('l1', 'l2'),
       'clf__C': (0.1, 1, 5),
 }
# 1. training_df_10w running
X_train = X_t
y_train = y_t
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,
   verbose=1, scoring='roc_auc', cv=3)
grid_search.fit(X_train, y_train)
print('Best score: %0.3f' % grid_search.best_score_)
print('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))
print('Precision:', precision_score(y_test, predictions))
print('Recall:', recall_score(y_test, predictions))

filename='SBdata'
training_edges = load_obj(filename)

ps_edges_for_node2vec=[]
for training_edge in training_edges:
    label = int(training_edge[1])
    source = int(training_edge[0][0])
    sink = int(training_edge[0][1])
    if label == 1:
        ps_edges_for_node2vec.append((source, sink))

import csv
filename='ps_edges_for_node2vec_sb_data'
with open(filename + ".txt", 'w', encoding = 'utf8') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(ps_edges_for_node2vec)

import pandas as pd


def get_common_neighbours(node1, node2):
    try:
        n1 = pre_features[node1]
        n2 = pre_features[node2]
        common_neighors = list(set(n1[2]).intersection(n2[2]))
        return common_neighors
    except:
        return []


training_df = pd.DataFrame()

# prepare dataset for ANN
for edge in final_edges:
    #     print(source, sink, label)
    source = edge[0]
    sink = edge[1]
    label = edge[2]
    common_neighbours = get_common_neighbours(source, sink)
    if len(common_neighbours) > 0:
        print('hello')
        print(len(common_neighbours))
    if len(common_neighbours) >= 100:
        common_neighbours = common_neighbours[:100]

    else:
        pass
    #     print(common_neighbours)
    common_neighbours.append(label)
    row_df = pd.DataFrame(common_neighbours)
    training_df = training_df.append(row_df)

filename='edges_10w'
with open(filename + ".txt", 'w', encoding = 'utf8') as f:
    f_csv = csv.writer(f)
    f_csv.writerows()