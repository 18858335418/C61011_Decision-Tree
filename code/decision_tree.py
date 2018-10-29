import math
import csv
import numpy as np
from collections import Counter
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import pandas as pd
import sys

sys.setrecursionlimit(1000000)

#
# def createDataset():
#     return data, feature


# calculate the information gain of the dateset
def calc_Etp(data):
    # put the data of features in the list
    features = Counter()
    features.update(data)
    data_num = len(data)

    entropy = 0.0
    for feature in features.values():
        # calc H(x)
        p = float(feature) / data_num
        entropy -= p * math.log2(p)

    return entropy


# calculate H(Y/X)
def calc_condition_etp(feature_list, label_list):
    entropy_dict = defaultdict(list)
    for i, value in zip(feature_list, label_list):
        entropy_dict[i].append(value)

    entropy = 0.0
    feature_num = len(feature_list)

    for value in entropy_dict.values():
        p = len(value) / feature_num * calc_Etp(value)
        entropy += p

    return entropy


# calculate information gain
def calc_infogain(feature_list, label_list):
    return calc_Etp(label_list) - calc_condition_etp(feature_list, label_list)


# now we inplement the function of building the desicion tree

# find the end of tree, judge if it should be splitted
def split(label_list):
    result = Counter(label_list)
    return len(result) == 1


# find the best feature for the branch of tree

def find_best_feature(dataset, labels, labelled_index):
    infogain_dict = {}  # store the information gain
    feature_num = len(dataset[0])
    for i in range(feature_num):
        if i in labelled_index:  # exclude the feature which has been in the leaves
            continue
        feature_list = [data[i] for data in dataset]
        infogain_dict[i] = calc_infogain(feature_list, labels)
        # print(infogain_dict[i])
    # find the feature with the biggest information gain
    feature = sorted(infogain_dict.items(), key=lambda calc_infogain: calc_infogain[1], reverse=True)
    # print(feature)
    return feature[0][0]


# define the Node class to store the node information while buliding the tree
class DecisionTreeNode(object):
    def __init__(self, dataset, labels, col=-1, predict_results=None, sub_node=[],feature_name=None,sub_index=[]):
        self.labelled_index = []  # labelled features
        self.dataset = dataset  # dataset using to build tree
        self.col = col  # the sort number of features
        self.labels = labels  # feature name for the node
        self.sub_node = sub_node
        self.predict_results = predict_results
        self.feature_name = feature_name
        self.sub_index = sub_index

class DecisionTree():
    def __init__(self):
        self.featurenum = 0
        self.tree_root = None

    ##bulid the desicion tree
    def build_tree(self, node: DecisionTreeNode):
        if split(node.labels):  # all examples have the same label
            print("all examples have the same label")
            node.predict_results = node.labels[0]
            print(node.predict_results)
            return
        index = find_best_feature(node.dataset, node.labels, node.labelled_index)
        node.col = index
        print(index)
        f_list=[]

        for i, value in enumerate(node.dataset):
            f_list.append(value[index])              # add all values into f_list
        print(f_list)
        feature_count = Counter(f_list)              #store in Counter
        print(feature_count)
        sub_index_list = []
        for j,keys_feature in enumerate(feature_count.keys()):
            sub_index = [i for i, value in enumerate(node.dataset) if value[index]==keys_feature]
            print(sub_index)
            sub_set = [node.dataset[i] for i in sub_index]
            sub_labels = [node.labels[i] for i in sub_index]
            sub_node = DecisionTreeNode(dataset=sub_set, labels=sub_labels)
            sub_node.labelled_index = list(node.labelled_index)
            sub_node.labelled_index.append(index)
            sub_node.feature_name=keys_feature
            sub_node.sub_index = sub_index
            node.sub_node.append(sub_node)

        for i in range(node.sub_node.__len__()):   #bug, buneng  digui
            print(i)
            if node.sub_node[i].sub_index:
                self.build_tree(node.sub_node[i])

    def fit(self, X, y):
        self.featurenum = len(X[0])
        self.tree_root = DecisionTreeNode(dataset=X, labels=y)
        self.build_tree(self.tree_root)

    def _predict(self, testdata, node: DecisionTreeNode):
        if node.predict_results:
            return node.predict_results
        no = node.col
        num = node.sub_node.__len__()
        for i in range(num):
            if testdata[no]==node.sub_node[i].feature_name:
                return self._predict(testdata, node.sub_node[i])

    def predict(self, testdata):
        return self._predict(testdata, self.tree_root)


# file = open("/Users/patrick/Documents/foundations of machine learning/lab/coursework/dataset/lymphography/lymphography.data.csv", "r")
file = open("/Users/patrick/Documents/foundations of machine learning/lab/coursework/dataset/Balloons/yellow-small+adult-stretch_number.data.csv", "r")
# df = pd.read_csv(file)
# print(df)
# af = df.drop_duplicates(['color', 'size', 'act', 'age'])
# af.to_csv("/Users/patrick/Documents/foundations of machine learning/lab/coursework/dataset/Balloons/yellow-small+adult-stretch_number.data copy1.csv",index = False)

reader = csv.reader(file)
headers = next(reader)

feature_list = []
label_list = []

for everyrow in reader:
    label_list.append(everyrow[-1])
    row = {}
    for x in range(0, len(everyrow) - 1):
        row[headers[x]] = int(everyrow[x])
    feature_list.append(row)

file.close()

print(label_list)
print(feature_list)

# extract the feature of train_X
vec = DictVectorizer()
data_X = vec.fit_transform(feature_list).toarray()
print(data_X)

lb = preprocessing.LabelBinarizer()
data_y = lb.fit_transform(label_list)
print(label_list)
print(data_y)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(data_X, data_y)


tree = DecisionTree()
tree.fit(data_X, label_list)
# for i in range(0,13):
#     first_row = data_X[i, :]
#     new_row = list(first_row)
#
#     print(data_X[i])
#     print(new_row)
#
#     print('predict:',tree.predict(new_row))
#     print('predict:',clf.predict([new_row]))