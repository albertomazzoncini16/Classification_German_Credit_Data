import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# %% set column names
attributes=['checking_balance',
           'months_loan_duration',
           'credit_history',
           'purpose',
           'amount',
           'savings_balance',
           'employment_duration',
           'installment_rate_income',
           'status_gender',
           'debtors_guarantors',
           'residence_years',
           'property',
           'age',
           'other_installment',
           'housing',
           'existing_loans_count',
           'job',
           'dependents',
           'phone',
           'class']

# %% load the data
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
url ='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
credit = pd.read_csv(url, sep=' ',header=None, names=attributes, index_col=False)
# %% Split the data
X=credit.drop('class', axis=1)
y=credit['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# check class balance
y.value_counts()/len(y)
y_train.value_counts()/len(y_train)
y_test.value_counts()/len(y_test)

# %% Calss to extract gender and status features

class AddGenderStatus(TransformerMixin, BaseEstimator):
    
    """This class extracts features from feature status_sex:
          A91 : male   : divorced/separated
          A92 : female : divorced/separated/married
          A93 : male   : single
          A94 : male   : married/widowed
          A95 : (female : single - does not exist) 
          """
          
    def __init__(self, key):
        # key is the column name as str
        self.key = key
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
            function_gender = lambda x:'male'if x=='A91'or x=='A93'or x=='A94' else 'female'
            function_status = lambda x: 'divorced' if x=='A91' else ('married' if x=='A92' or x=='A94' else 'single')
            X_new = X.copy()
            X_new["status"] = X[self.key].map(function_status)
            X_new["gender"] = X[self.key].map(function_gender)
            X_new.drop([self.key], axis=1,inplace=True)
            return X_new

# %% Pipeline new_attribs
gender_status_attribs = Pipeline([
                        ('AddGenderStatus',AddGenderStatus(key='status_gender'))
                        ])
# %% Small check
# X_train_check = gender_status_attribs.transform(X_train)
# 'gender' and 'status' in list(X_train_check) #True
# %% Create a class to select numerical or categorical columns 

class ColumnExtractor(BaseEstimator,TransformerMixin):
    def __init__(self, key):
        # key is the column name as str
        self.key = key
    def fit(self, X, y=None):
        # stateless transformer
        return self
    def transform(self, X):
        # assumes X is a DataFrame
        return X[self.key]

# %% Preprocessing categorical an numerical data

cat_attribute=[]
num_attribute=[]
for col in X.columns:
    # print(credit[col].dtype)
    if X[col].dtype == 'object':
        cat_attribute.append(col)
    else:
        num_attribute.append(col)
 
new_attributes = ['status','gender']   
cat_attribute = cat_attribute + new_attributes
del cat_attribute[5]

categorical_attribs_trans = ColumnTransformer([
    ('encoder',OneHotEncoder(drop='first',sparse=False),cat_attribute),
    ('scaler', StandardScaler(), num_attribute)
    ])


# %% Preprocessing Pipeline

preprocessing_pipeline = Pipeline([("gender_status_attribs", gender_status_attribs),
                          ("categorical_attribs", categorical_attribs_trans),
                                               ])

X_train=preprocessing_pipeline.fit_transform(X_train)
# %% Preprocessing dependent variable 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train=encoder.fit_transform(y_train) # 0=yes 1=no
# %% Models
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

svc_clf=SVC(kernel="rbf", gamma=5, C=0.001)
svc_clf.fit(X_train, y_train)

dt_clf=DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

rf_clf=RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

ada_clf = AdaBoostClassifier( DecisionTreeClassifier())
ada_clf.fit(X_train, y_train)

clf_models = [sgd_clf,knn_clf,svc_clf,dt_clf,rf_clf]
clf_models_names = ['SGD','KNN','SVC','Decision_tree','Random_forest']
# %% Performance Measures
# %% Cross Validation
from sklearn.model_selection import cross_val_score
"""
K-fold cross-validation means splitting the training set into K-folds, 
then making predictions and evaluating them on each fold using a model trained 
on the remaining folds.
"""
for model in clf_models:
    for name in clf_models_names:
        print(name, cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean())
#  array([0.928, 0.96 , 0.932]) ratio of correct predictions
# SGD 0.962
# KNN 0.964
# SVC 0.964
# Decision_tree 0.964
# Random_forest 0.964

# Build a classifier that classify all as 0 = yes 
class BaseClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1))
 

unique, counts = np.unique(y_train, return_counts=True)
counts[0]/len(y_train) # 96.4% of the class is 0 

cross_val_score(BaseClassifier(), X_train, y_train, cv=3, scoring="accuracy")
# array([0.972, 0.972, 0.948])
cross_val_score(BaseClassifier(), X_train, y_train, cv=3, scoring="accuracy").mean()  # 96.4%
"""
This demonstrates why accuracy is generally not the preferred performance 
measure for classifiers, especially when you are dealing with skewed datasets 
(i.e., when a classes is much more frequent).
""" 
# %% Confusion Matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
# SGS
y_train_pred_sgd = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
confusion_matrix(y_train_pred_sgd,y_train)
# array([[701,  22],
#        [ 23,   4]])
# KNN
y_train_pred_knn = cross_val_predict(knn_clf, X_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred_knn)
# array([[723,   0],
#        [ 27,   0]])
# SVC
y_train_pred_SVC = cross_val_predict(svc_clf, X_train, y_train, cv=3)
confusion_matrix(y_train_pred_SVC,y_train)
# array([[723,  27],
#        [  0,   0]])
# DecisionTree
y_train_pred_dt=cross_val_predict(dt_clf,X_train,y_train, cv=3)
confusion_matrix(y_train_pred_dt,y_train)
# array([[692,  19],
#        [ 31,   8]])
# RandomForest
y_train_pred_rf=rf_clf.predict(X_train) #no cross validation randomforest is an ensamble model
confusion_matrix(y_train, y_train_pred_rf)
# array([[723,   0],
#        [  0,  27]])
# we could stop here! perfect on the train set!but wait the result on the test set.
# Adaboost
y_train_pred_ada=ada_clf.predict(X_train) #no cross validation randomforest is an ensamble model
confusion_matrix(y_train, y_train_pred_ada)
# array([[723,   0],
       # [  0,  27]])

# %% Precision and Recall + precision/recall tradeoff
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

# SGS
precision_score(y_train, y_train_pred_sgd) # 15.38% precision
recall_score(y_train, y_train_pred_sgd)    # 14.81% recall
"""
When the model claims a class is 1, it is correct only 15.38% of the time, and it only detects 14.81% of the 1s.
"""
# The f1_score is the harmonic mean of precision and recall
f1_score(y_train, y_train_pred_sgd)        # 0.1509433962264151


# DecisionTree
precision_score(y_train, y_train_pred_dt) # 20.5% precision
recall_score(y_train, y_train_pred_dt)    # 29.6% recall
f1_score(y_train, y_train_pred_dt)        # 0.242

# RandomForest
precision_score(y_train, y_train_pred_rf) # 100% precision
recall_score(y_train, y_train_pred_rf)    # 100% recall
f1_score(y_train, y_train_pred_rf)        # 1

# Increasing precision reduces recall, and vice versa. This is called the precision/recall tradeoff.
"""
For each instance, SGDClassifier computes a score based on a decision function,
and if that score is greater than a threshold, it assigns the instance to the
positive class, or else it assigns it to the negative class.
.decision_function() method returns a score for each instance,
and then make predictions based on those scores using any threshold.
The SGDClassifier uses a threshold equal to 0, raising the threshold decreases recall.
"""
 
# y_scores_sgd = sgd_clf.decision_function(X_train)
# threshold = 200000
# y_train_pred = (y_scores_sgd > threshold)

# %% Plot Precision and recall versus the decision threshold
y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function") # decision funct because is a buil-in in SGD
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_sgd)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
    plt.xlabel("Threshold")
    plt.title('Precision and recall versus the decision threshold')
    plt.legend() 
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds) 
# recall precision curve
plt.plot(recalls, precisions) 
plt.xlabel("recall")
plt.ylabel("precision")

# %% The ROC Curve - receiver operating characteristic curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# SGD
y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function") 
# decision funct because is a buil-in in SGD
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train, y_scores_sgd)
roc_auc_score(y_train, y_scores_sgd) # 0.6061164899339173
# KNN
y_probas_knn = cross_val_predict(knn_clf, X_train, y_train, cv=3, method="predict_proba") 
y_scores_knn = y_probas_knn[:, 1]
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_train,y_scores_knn)
roc_auc_score(y_train, y_scores_knn) # 0.6187951436914092
# SVC
y_scores_svc = cross_val_predict(svc_clf, X_train, y_train, cv=3, method="decision_function") 
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_train,y_scores_svc)
roc_auc_score(y_train, y_scores_svc) # 0.6904359407817223
# DecisionTree
y_probas_dt = cross_val_predict(dt_clf, X_train, y_train, cv=3, method="predict_proba") 
y_scores_dt = y_probas_dt[:, 1]
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_train,y_scores_dt)
roc_auc_score(y_train, y_scores_dt) #0.6232518825879821
# RandomForest
y_probas_rf = cross_val_predict(rf_clf, X_train, y_train, cv=3,method="predict_proba") 
# RandomForestClassi fier class does not have a decision_function
y_scores_rf = y_probas_rf[:, 1] # score = proba of positive class
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_train,y_scores_rf)
roc_auc_score(y_train, y_scores_rf) # 0.7656370063009068

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label) 
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])
    plt.title('The ROC Curve')
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    

plot_roc_curve(fpr_sgd, tpr_sgd,label="SGD")
plot_roc_curve(fpr_knn, tpr_knn,label="KNN")
plot_roc_curve(fpr_svc, tpr_svc,label="SVC")
plot_roc_curve(fpr_dt, tpr_dt,label="DecisionTree")
plot_roc_curve(fpr_rf, tpr_rf, "RandomForest")
plt.legend()

# %% GridSearch on RandomForest - so far the best model
from sklearn.model_selection import GridSearchCV
np.arange(50,500,10)

param_grid = [{'n_estimators': np.arange(200,500,25), 'max_features': np.arange(5,50,5)},
              {'bootstrap': [False], 'n_estimators': np.arange(200,500,25), 'max_features': np.arange(5,50,5)},
              ]

grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_

RandomForestClassifier(bootstrap=False, max_features=10, n_estimators=200,
                       random_state=42)

feature_importances = pd.DataFrame(grid_search.best_estimator_.feature_importances_)
feature_importances['feature_number']=np.arange(1,len(feature_importances)+1,1)
feature_ranking = feature_importances.sort_values(0,ascending=False)


param_grid_ada =[{'n_estimators': np.arange(100,400,25)}
                 
    ]
grid_search_ada = GridSearchCV(ada_clf, param_grid_ada, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search_ada.fit(X_train, y_train)
grid_search_ada.best_estimator_
# %% The test set
y_test=encoder.fit_transform(y_test) 
X_test=preprocessing_pipeline.fit_transform(X_test)

rf_clf_tuned = RandomForestClassifier(bootstrap=False, max_features=10, n_estimators=200,
                       random_state=42)
rf_clf_tuned.fit(X_train,y_train)
y_test_pred = rf_clf_tuned.predict(X_test)

confusion_matrix(y_test, y_test_pred)
# array([[239,   1],
#        [  9,   1]])

(y_test == 1).sum() # 10
(y_test == 0).sum() # 240
 
# |:( no good

ada_clf_tuned=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=300)
ada_clf_tuned.fit(X_train,y_train)
y_test_pred_ada = ada_clf_tuned.predict(X_test)
confusion_matrix(y_test, y_test_pred_ada)
# array([[230,  10],
#        [  9,   1]])







