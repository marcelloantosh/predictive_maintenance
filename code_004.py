# Import basic dependencies
import pandas as pd

from sklearn.metrics import average_precision_score, accuracy_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler


from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, KFold


# import data
df_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv')

# Preview the first few rows of data
df_raw.head()

# Get basic information about the data
df_raw.info()

# Check columns for unique values
df_raw.nunique()

# Drop identifier and particular failure type columns, but not 'Machine failure'
df_dropped_columns_most = df_raw.drop(['UDI', 'Product ID', 'TWF','HDF','PWF','OSF','RNF'],axis=1)

# Create separate df also dropping 'Machine failure'
df_dropped_columns_all = df_dropped_columns_most.drop(['Machine failure'],axis=1)

# Create X dataframe by performing one-hot encoding on df_dropped_columns
X = pd.get_dummies(df_dropped_columns_all)

# Create target y dataframe
y = df_raw['Machine failure']

# Preview X
X.head() 

# Preview y
y.head() 

# create a MinMaxScaler object
scaler_normalized = MinMaxScaler()

# normalize the data using the scaler
X_normalized = pd.DataFrame(scaler_normalized.fit_transform(X), columns=X.columns)

# replace square brackets in column names to avoid ValueError with XGB model below
X_normalized.columns = X_normalized.columns.str.replace('\[','(').str.replace('\]',')')

# preview normalized data
X_normalized.head()

# Evaluate multiple models at once

# create dictionary of model names and models
models = {"Logistic Regressor": LogisticRegression(),
          "Random Forest Classifier": RandomForestClassifier(),
          "Ada Boost Classifier": AdaBoostClassifier(),
          "XGB Classifier": XGBClassifier()}

# create dictionary of evaluation metrics to compute
scoring = {'average_precision': 'average_precision', 
           'recall': 'recall', 
           'accuracy': 'accuracy', 
            'f1': 'f1'}

# initialize empty lists to store metrics and model names
aps_scores = []
recall_scores = []
accuracy_scores = []
f1_scores = []
model_names = []

# split the data into train, test sets for initial fitting
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=1, stratify=y)

# Loop through each model to fit and evaluate
for model_name, model in models.items():
    
    # Fit models on the training data
    model.fit(X_train, y_train)

    # make preditions on test set, calculate probabilities on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    
    # compute set of metrics to evaluate, compare models
    aps = average_precision_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Append scores and model name to respective lists
    aps_scores.append(aps)
    recall_scores.append(recall)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    model_names.append(model_name)

    # Print the evaluation metrics for the test data
    print(f"{model_name}:\nAverage Precision Score: {aps}\nRecall Score: {recall}\nAccuracy Score: {accuracy}\nF1 Score: {f1}\n")

    # Perform stratified k-fold cross-validation on the model and print the results
    skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    skf_cv_results = cross_validate(model, X_normalized, y, cv=skf_cv, scoring=scoring, return_train_score=False)
    print(f"{model_name} Stratified K-Fold Cross-Validation Metrics:")
    for metric in scoring:
        mean_skf_cv_score = skf_cv_results[f"test_{metric}"].mean()
        print(f"{metric.capitalize()} Score: {mean_skf_cv_score}")
    print()

