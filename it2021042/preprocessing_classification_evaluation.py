import itertools
from itertools import chain, combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori, association_rules
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.feature_selection import RFE, SelectFromModel, RFECV


# ------------------- Α) Προετοιμασία (10%) -------------------

# --------- Part 1a - Προετοιμασία Δεδομένων ---------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # print(df.shape)
    df.replace('?', np.nan, inplace=True)
    # print(df.info())
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,'percent_missing': percent_missing})
    missing_value_df = missing_value_df[missing_value_df['percent_missing'] > 0]
    missing_value_df = missing_value_df.sort_values(by='percent_missing', ascending=False)
    # columns_to_drop = missing_value_df.head(2)['column_name'].values # drop the 2 highest
    # slightly worse performance
    # df = df.drop(columns=columns_to_drop)
    # print(missing_value_df)

    non_numeric_cols = df.select_dtypes(include=['object']).columns
    # print(non_numeric_cols)

    for col in non_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # print(f"Column: {col}")
        # print(f"Unique Values: {df[col].unique()}\n")


    # non_numeric_cols = df.select_dtypes(include=['object']).columns
    # print(non_numeric_cols)
    # print(df.info())

    df = df.fillna(df.mean(numeric_only=True))
    return df



df = pd.read_csv('training_companydata.csv')
df = clean_dataframe(df)

# ------------------- Β) Κατηγοριοποίηση (60-70%) -------------------

# --------- Part 1b - Αξιολόγηση Μοντέλου ---------

def evaluate_model(X, y, get_imp = False):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    conf_matrix_sum = np.zeros((2, 2), dtype=int)
    accuracies = []
    precision_values = []
    recall_values = []
    f1_values = []
    conf_matrix_sum_2 = np.zeros((2, 2), dtype=int)
    accuracies_2 = []
    precision_values_2 = []
    recall_values_2 = []
    f1_values_2 = []
    importances = []
    for train_index, test_index in kf.split(X, y):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        smote = SMOTE(random_state=0)
        Xs_train, ys_train = smote.fit_resample(X_train, y_train)

        # Κανονικοποιούμε τα δεδομένα στο [0, 1]
        scaler = MinMaxScaler()
        scaler.fit(Xs_train)
        Xn_train = scaler.transform(Xs_train)
        Xn_test = scaler.transform(X_test)


        # Εκπαιδεύoυμε ένα μοντέλο λογιστικής παλινδρόμησης (scikit-learn)
        # model = LogisticRegression(random_state=0)
        # model = RandomForestClassifier(random_state=0)
        # model = DecisionTreeClassifier(random_state=0)
        # model = DecisionTreeRegressor(random_state=0)
        # model = NearestCentroid()
        # model = GaussianNB()
        # model = SVC(random_state=0)
        model = xgb.XGBClassifier(random_state=0, early_stopping_rounds=5, eval_metric="logloss")


        model.fit(Xn_train, ys_train, eval_set=[(Xn_test, y_test)], verbose=0)

        y_hat_test_prob = model.predict_proba(Xn_test)[:, 1]
        y_hat_test = (y_hat_test_prob >= 0.5).astype(int)

        y_hat_train_prob = model.predict_proba(Xn_train)[:, 1]
        y_hat_train = (y_hat_train_prob >= 0.5).astype(int)
        # print(y_test, y_hat_test)

        # Αξιολογούμε την ευστοχία στο σύνολο δοκιμής και στο σύνολο εκπαίδευσης
        if get_imp == True:
            importances.append(model.feature_importances_)
        else:
            fold_conf_matrix = confusion_matrix(y_test, y_hat_test)
            accuracy = accuracy_score(y_test, y_hat_test)
            precision = precision_score(y_test, y_hat_test)
            recall = recall_score(y_test, y_hat_test)
            f1 = f1_score(y_test, y_hat_test)

            fold_conf_matrix_2 = confusion_matrix(ys_train, y_hat_train)
            accuracy_2 = accuracy_score(ys_train, y_hat_train)
            precision_2 = precision_score(ys_train, y_hat_train)
            recall_2 = recall_score(ys_train, y_hat_train)
            f1_2 = f1_score(ys_train, y_hat_train)

            conf_matrix_sum += fold_conf_matrix
            accuracies.append(accuracy)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

            conf_matrix_sum_2 += fold_conf_matrix_2
            accuracies_2.append(accuracy_2)
            precision_values_2.append(precision_2)
            recall_values_2.append(recall_2)
            f1_values_2.append(f1_2)
            # print("Model accuracy on test set:", accuracy)
            # print(y_hat_test_prob)

    if get_imp == True:
        mean_importances = np.mean(importances, axis=0)
        top_10_indices = np.argsort(mean_importances)[-10:][::-1]

        # Get the top 10 features and their importances
        top_10_features = [(X.columns[i], mean_importances[i]) for i in top_10_indices]

        print("Top 10 features with their importances and positions:")
        for (feature, importance) in zip(top_10_indices, top_10_features):
            print(f"Mean Importance: {importance}")

        X_imp = X.iloc[:, top_10_indices]
        return X_imp

    avg_conf_matrix = conf_matrix_sum / kf.get_n_splits()
    avg_conf_matrix = np.round(avg_conf_matrix).astype(int)
    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precision_values)
    average_recall = np.mean(recall_values)
    average_f1_score = np.mean(f1_values)

    avg_conf_matrix_2 = conf_matrix_sum_2 / kf.get_n_splits()
    avg_conf_matrix_2 = np.round(avg_conf_matrix_2).astype(int)
    average_accuracy_2 = np.mean(accuracies_2)
    average_precision_2 = np.mean(precision_values_2)
    average_recall_2 = np.mean(recall_values_2)
    average_f1_score_2 = np.mean(f1_values_2)

    print("\n-------- Predictions On Testing Data --------")
    print("Average Confusion Matrix:\n", avg_conf_matrix)
    print(f'Average Accuracy: {average_accuracy}')
    print(f"Average Precision: {average_precision}")
    print(f"Average Recall: {average_recall}")
    print(f"Average F1-Score: {average_f1_score}")

    print("\n-------- Predictions On Training Data --------")
    print("Average Confusion Matrix:\n", avg_conf_matrix_2)
    print(f'Average Accuracy: {average_accuracy_2}')
    print(f"Average Precision: {average_precision_2}")
    print(f"Average Recall: {average_recall_2}")
    print(f"Average F1-Score: {average_f1_score_2}")



X = df.drop('X65', axis=1)
y = df['X65']
evaluate_model(X, y)


# --------- Part 2 - Πρόβλεψη στα άγνωστα δεδομένα ---------
X_known = X
y_known = y

df2 = pd.read_csv('test_unlabeled.csv')
# Extract the current column names and store them in the first row
previous_columns = df2.columns.tolist()

# Define your new column names (e.g., X1, X2, ..., XN)
new_columns = ['X' + str(i) for i in range(1, df2.shape[1] + 1)]

# Insert the previous column names as the first row
df2.loc[-1] = previous_columns  # Insert the old column names as the first row
df2.index = df2.index + 1  # Shift the index by 1 to make space for the new row
df2 = df2.sort_index()  # Sort the index to push the first row down

# Assign new column names
df2.columns = new_columns
# print(df2)
df2 = clean_dataframe(df2)
X_unknown = df2

smote = SMOTE(random_state=0)
Xs_known, ys_known = smote.fit_resample(X_known, y_known)

# print(Xs_known.columns)
# print(X_unknown.columns)

# Κανονικοποιούμε τα δεδομένα στο [0, 1]
scaler = preprocessing.MinMaxScaler()
scaler.fit(Xs_known)
Xn_known = scaler.transform(Xs_known)
Xn_unknown = scaler.transform(X_unknown)

model = xgb.XGBClassifier(random_state=0)
model.fit(Xn_known, ys_known)
y_hat_prob = model.predict_proba(Xn_unknown)[:, 1]
y_hat = (y_hat_prob >= 0.5).astype(int)

with open('predictions.txt', 'w') as f:
    for prediction in y_hat:
        f.write(f"{prediction}\n")

# --------- Part 4 - 50 εταιρίες που φαίνεται πιθανότερο να χρεωκοπήσουν ---------

# Get probabilities and indices
bankruptcy_candidates = [(index, prob) for index, prob in enumerate(y_hat_prob) if prob >= 0.5]

# Sort by probabilities in descending order
sorted_candidates = sorted(bankruptcy_candidates, key=lambda x: x[1], reverse=True)

# Select top 50 candidates
top_50_candidates = sorted_candidates[:50]

# Write to file
with open('top_50_bankruptcies.txt', 'w') as f:
    for row_id, prob in top_50_candidates:
        f.write(f"{row_id+1}\n")


# ------------------- Γ) Αξιολόγηση Γνωρισμάτων - Παλινδρόμηση (30%) -------------------

# --------- Part 3 - Υποσύνολα Γνωρισμάτων ---------

# To RFECV ειναι το πιο σωστο για να παρουμε 1 subset γνωρισματων απο ολα τα kfolds, αλλα ειναι πολυ computationally heavy

# rfe = RFECV(estimator=model, min_features_to_select=10, verbose=1)  # Select top 10 features
# X_rfe = rfe.fit_transform(X, y)
# max_features = 10
# top_features_indices = feature_ranking.argsort()[:max_features]  # Get indices of the top features
# selected_columns = X.columns[top_features_indices]
# X_rfe = X.loc[:, selected_columns]       # Filter the DataFrame
# # X_rfe = X[:, selector.support_]
# # X_rfe = pd.DataFrame(X_rfe, columns=selected_columns)
# print(X_rfe)

X_imp = evaluate_model(X, y, get_imp=True)
evaluate_model(X_imp, y)

# exoume xeirotero performance me mono 10 features
