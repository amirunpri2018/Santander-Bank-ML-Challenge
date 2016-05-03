import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import MiniBatchKMeans

print "Importing Files..."

df = np.genfromtxt ('train_normal_286.csv', delimiter=",")
X = df[:, 0:286]
y = df[:, 286]

print "Counting Zeros..."

ncounts = np.zeros((X.shape[0], 1))
for i in range(0, X.shape[0]):
	ncounts[i, 0] = (X[i, :] == 0).sum(0)
X = np.append(X, ncounts, axis = 1)
#print X[1, -1]

print "Clustering the train set..."

clusters = MiniBatchKMeans(n_clusters = 2, max_iter = 100, batch_size = 3008)
clusters.fit(X)
categories = clusters.predict(X)
cats = np.zeros((len(categories), 1))
for i in range(0, cats.shape[0]):
	cats[i, 0] = categories[i]
X = np.append(X, cats, axis = 1)
#print X[1, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=11)

print "Defining models..."

#####Defining All Model Parameters
ada = AdaBoostClassifier(n_estimators = 92, learning_rate = 0.32, random_state = 12)
dt = DecisionTreeClassifier(max_depth = 8)
rf = RandomForestClassifier(max_depth = 9, n_estimators = 25, max_features = 21, verbose = 0)
bagged_dt = BaggingClassifier(base_estimator = dt, n_estimators = 10, random_state = 12, verbose = 0)
bagged_rf = BaggingClassifier(base_estimator = rf, n_estimators = 20, random_state = 12, verbose = 0)
calibrated_dt = CalibratedClassifierCV(bagged_dt, method = 'isotonic', cv = 4)
calibrated_rf = CalibratedClassifierCV(bagged_rf, method = 'isotonic', cv = 4)

#######Training all models

print "Fitting AdaBoost..."
ada.fit(X_train, y_train)
print "adaboost test:",roc_auc_score(y_test, ada.predict_proba(X_test)[:,1])
#print "adaboost train:",roc_auc_score(y_train, ada.predict_proba(X_train)[:,1])

#print "Fitting Decision Tree..."
#dt.fit(X_train, y_train)
#print "Decision tree test:", roc_auc_score(y_test, dt.predict_proba(X_test)[:,1])
#print "Decision tree train:",roc_auc_score(y_train, dt.predict_proba(X_train)[:,1])

#print "Fitting Random Forest..."
#rf.fit(X_train, y_train)
#print "random forest test:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
#print "random forest train:", roc_auc_score(y_train, rf.predict_proba(X_train)[:,1])

print "Bagging Decision Trees..."
bagged_dt.fit(X_train, y_train)
print "bagged dt test:", roc_auc_score(y_test, bagged_dt.predict_proba(X_test)[:,1])
#print "bagged dt train",roc_auc_score(y_train, bagged_dt.predict_proba(X_train)[:,1])

print "Bagging RandomForests..."
bagged_rf.fit(X_train, y_train)
print "bagged rf test",roc_auc_score(y_test, bagged_rf.predict_proba(X_test)[:,1])
#print "bagged rf train",roc_auc_score(y_train, bagged_rf.predict_proba(X_train)[:,1])

'''print "Calibrating Bagged Decision Trees..."
calibrated_dt.fit(X_train, y_train)
print "calibrated_dt test:", roc_auc_score(y_test, calibrated_dt.predict_proba(X_test)[:,1])

print "Calibrating Bagged Random Forests..."
calibrated_rf.fit(X_train, y_train)
print "calibrated_rf test:", roc_auc_score(y_test, calibrated_rf.predict_proba(X_test)[:,1])
'''
print "Voting with all models...."
voted_model = VotingClassifier(estimators=[('one', ada), ('two', bagged_rf), ('four', bagged_dt)], voting='soft')
voted_model.fit(X_train, y_train)
print "Voted Model test:",roc_auc_score(y_test, voted_model.predict_proba(X_test)[:,1])
#print "Voted Model train",roc_auc_score(y_train, voted_model.predict_proba(X_train)[:,1])

####Loading test file and saving predictions

print "Saving Voted Submission"
X_test = np.genfromtxt ('test_normal_286.csv', delimiter=",")
ncounts = np.zeros((X_test.shape[0], 1))
for i in range(0, X_test.shape[0]):
	ncounts[i, 0] = (X_test[i, :] == 0).sum(0)
X_test = np.append(X_test, ncounts, axis = 1)

categories_test = clusters.predict(X_test)
cats = np.zeros((len(categories_test), 1))
for i in range(0, cats.shape[0]):
	cats[i, 0] = categories_test[i]
X_test = np.append(X_test, cats, axis = 1)


predictions = voted_model.predict_proba(X_test)[:, 1]
np.savetxt("Voted_Submission.csv", predictions, delimiter = ',')