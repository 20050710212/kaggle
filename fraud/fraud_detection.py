# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#

#
df = pd.read_csv('creditcard.csv')
df['Class'].value_counts()
# class0 284315, class1 492 使用 stratifi的分类方法
sns.countplot('Class',data = df)

#
from sklearn.preprocessing import RobustScaler, StandardScaler
robscaler = RobustScaler()
df['scaled_amount'] = robscaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = robscaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Amount','Time'], axis = 1, inplace = True)

#
from sklearn.model_selection import StratifiedKFold
X = df.drop('Class', axis = 1)
y = df['Class']

# 这里用 stratified对数据进行拆分，保证train和test里class0 和1的比例与原始数据一致
sss = StratifiedKFold(n_splits = 5, random_state = None, shuffle = False)
for train_index, test_index in sss.split(X,y):
    #print("length of train_index is", len(train_index))
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
##
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts = True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts = True)   
print(train_counts_label/len(original_ytrain))
print(test_counts_label/len(original_ytest))

# set frac = 1 is to shuffle the samples
df = df.sample(frac = 1)
fraud_df = df.loc[df['Class'] == 1]
nofraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, nofraud_df])
new_df = normal_distributed_df.sample(frac = 1, random_state = 42)
print(new_df['Class'].value_counts()/len(new_df))
sns.countplot('Class', data = new_df)
## correlation, 这里需要对两种情况进行分析，一个是没有重采样的，一个是重采样，比例为1：1的
f,(ax1, ax2) = plt.subplots(2,1,figsize = [24,20])

corr = df.corr()
sns.heatmap(corr, annot_kws = {'size' : 20}, ax = ax1)
ax1.set_title('Imbalanced correlation matrix', fontsize = 14)

new_corr = new_df.corr()
sns.heatmap(new_corr, annot_kws = {'size' : 20}, ax = ax2)
ax2.set_title('Balanced correlation matrix', fontsize = 14)
plt.show()
## 在imbalance的correlation中，几乎找不到正关系，但是在balanced中，
## V2,V4,V11,V19 是明显的正关系，V1，V3，V7，V10,V12,V14,V16,V17, 
## 选择其中几种先看下正关系, V19关系最弱
f, axes = plt.subplots(ncols = 4, figsize = [20,4])
sns.boxplot(x ='Class', y = 'V2', data = new_df, ax = axes[0])
axes[0].set_title('the positive correlation between V2 and y')

sns.boxplot(x ='Class', y = 'V4', data = new_df, ax = axes[1])
axes[1].set_title('the positive correlation between V4 and y')

sns.boxplot(x ='Class', y = 'V11', data = new_df, ax = axes[2])
axes[2].set_title('the positive correlation between V11 and y')

sns.boxplot(x ='Class', y = 'V19', data = new_df, ax = axes[3])
axes[3].set_title('the positive correlation between V19 and y')

plt.show()

#
f, axes = plt.subplots(ncols = 4, figsize = [20,4])

sns.boxplot(x = 'Class', y = 'V10', data = new_df, ax = axes[0])
axes[0].set_title('the negative correlation between V10 and class')
sns.boxplot(x = 'Class', y = 'V12', data = new_df, ax = axes[1])
axes[1].set_title('the negative correlation between V12 and class')
sns.boxplot(x = 'Class', y = 'V14', data = new_df, ax = axes[2])
axes[2].set_title('the negative correlation between V14 and class')
sns.boxplot(x = 'Class', y = 'V17', data = new_df, ax = axes[3])
axes[3].set_title('the negative correlation between V17 and class')
plt.show()

## 这里有一个inter quatile range 的去处outlier的过程，先不用。

##
X = new_df.drop(['Class'], axis = 1)
y = new_df['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

## 几种常见的分类器
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
classifiers ={ 'LRC': LogisticRegression(),
             'KNC': KNeighborsClassifier(),
             'SVC': SVC(),
             'DTC': DecisionTreeClassifier()
             }
for key, classifier in classifiers.items():
    classifier.fit(X_train,y_train)
    trainning_scores = cross_val_score(classifier, X_train,y_train, cv = 5)
    print('Classifiers:', key,'Has a trainning score,', classification_report(trainning_scores.mean(),2)*100, '% accuracy')
## 这个只是accuracy ，不重要,关键是precision，recall，f1score和auc
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import classification_report
y_train_pred = classifiers['LRC'].predict_proba(X_train)
##
X = df.drop('Class', axis = 1)
y = df['Class']
X_values = X.values
y_pred_prob = classifiers['LRC'].predict_proba(X_values)
y_pred = classifiers['LRC'].predict(X_values)
precision, recall, threshold = precision_recall_curve(y, y_pred_prob[:,1])
plt.figure(figsize = [12,6])
plt.plot(precision, recall)
log_reg_conf = confusion_matrix(y,y_pred)
sns.heatmap(log_reg_conf, annot = True, cmap = plt.cm.copper)
## 这个undersample的方法，能够得到很好的recall，但是precision不好。
len_class0, len_class1 = df.Class.value_counts()

def plotConfusionMatrixClassificationReport(y, y_pred, len_class0, len_class1):
    log_reg_conf = confusion_matrix(y,y_pred)
    denominator= np.array([[1/len_class0, 1/len_class0],[1/len_class1, 1/len_class1]])
    log_reg_conf_norm = log_reg_conf * denominator
    fig = plt.figure(figsize=[14,5])
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(log_reg_conf, annot = True, cmap = plt.cm.copper, ax = ax)
    plt.title('Confusion Matrix without normalization')
    plt.xlabel('Predicted Classes')
    plt.ylabel('Real Classes')
    
    ax = fig.add_subplot(1,2,2)
    sns.heatmap(log_reg_conf_norm, annot = True, cmap = plt.cm.copper, ax = ax)
    plt.title('Confusion Matrix with normalization')
    plt.xlabel('Predicted Classes')
    plt.ylabel('Real Classes')
    
    print('         -------- Classification Report --------')
    print(classification_report(y, y_pred))
plotConfusionMatrixClassificationReport(y, y_pred, len_class0, len_class1)
# 看这个图，recall 很高，0.92，漏判很低。而且误判看起来也很低，0.035，但是实际上的precision很低
# 不用undersample来处理 imbalance
df = pd.read_csv('creditcard.csv')
#df['hour'] = df['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)
#df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
X = df.drop('Class', axis = 1)
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(X,y, stratify = y, test_size = 0.35)
_,(len_class0, len_class1) = np.unique(y_test, return_counts = True)
lr_model = LogisticRegression(class_weight = 'balanced')
lr_model.fit(x_train,y_train)
y_pred = lr_model.predict(x_test)
plotConfusionMatrixClassificationReport(y_test, y_pred, len_class0, len_class1)

for w in [1,5,10,50,100,500]:
    print('weight is {} for fraud class --'.format(w))
    lr_model = LogisticRegression(class_weight = {0:1,1:w})
    lr_model.fit(x_train,y_train)
    y_pred = lr_model.predict(x_test)
    plotConfusionMatrixClassificationReport(y_test, y_pred, len_class0, len_class1)
##
fig = plt.figure(figsize = [15,8]) 
ax1 = fig.add_subplot(1,2,1)
ax1.set_title('ROC CURVE')
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
plt.grid()

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('PR CURVE')
ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([-0.05, 1.05])
ax2.set_xlabel('recall')
ax2.set_ylabel('precision')
plt.grid()
for w,k in zip([1,5,10,20,50,100,10000], 'bgrcmykw'):
    lr_model = LogisticRegression(class_weight = {0:1,1:w})
    lr_model.fit(x_train,y_train)
    y_pred = lr_model.predict(x_test)
    y_pred_prob = lr_model.predict_proba(x_test)[:,1]
    p,r,_ = precision_recall_curve(y_test, y_pred_prob)
    fpr, tpr,_ = roc_curve(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred_prob)
    pr_score = auc(r,p)
    print('weight is {} for fraud class --'.format(w))
    print('the precision score is,',precision_score )
    print('the recall score is,',recall_score )
    print('the pr score is,', pr_score)
    ax1.plot(fpr, tpr, c=k, label = w)
    ax2.plot(r,p, c = k, label = w)
ax1.legend(loc = 'lower right')
ax2.legend(loc = 'lower left')
plt.show()
    
## SMOTE
from imblearn.over_sampling import SMOTE


