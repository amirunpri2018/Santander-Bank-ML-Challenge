# Santander-Bank-ML-Challenge
Contains all the ML code used to achieve 117nd place on the public LB on kaggle's Santander Customer Satisfaction Challenge.

The file R file is the public script for the tuned XGB parameters to run on the original train and test files of the challenge. The hard coded rules of setting certain values below a certain threshold to 0 clearly won't generalize well, but at the end, all I was trying to do was overfit the public LB as much as I could.

The Python Script uses many different models, and even uses clustering as a feature to increase the ROC AUC metric to achieve a higher score on the public LB. You will notice there is no cross validation, since the time required to do stratified 5-fold CV mutliple times, throwing out best and worst folds, and performing other such operations would've been very time consuming, I used a small ROC check and uploaded to kaggle alot to check the LB score. You will notice there are alot of models in there which are commented out, and those were used for testing, you can play around with them if you want.

We play around with bagged random forests and decision trees, adaboost, clustering (Mini Batch K means), model calibration and model voting. Logistic Regression was tried in Octave, didn't work too well for this particular dataset.

The final submission was made by averaging the result of the XGBOOST model, and the Python script (0.95*XGB + 0.05*Python)/2.

Although the public LB score was great, what was suspected really did happen, the model was overfitted, and without any CV, there was no way of knowing, and the score on the final private LB dropped to 2500 approx.

Lessons learned:
1. DONT OVERFIT THE PUBLIC LB! No matter how tempting.
2. Always use K - fold CV to check your model's bias and variance. 
3. Unbalanced data sets are hard to use with Neural Nets, try SMOTE.
4. SVMs are really slow, need a better computer to train an svm on a large data set.
5. Voting Models may work well in some cases.


