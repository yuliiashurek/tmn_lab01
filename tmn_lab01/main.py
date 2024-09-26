# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, make_scorer, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Load the dataset from the uploaded CSV file
file_path = 'data/bioresponse.csv'
data = pd.read_csv(file_path)

# Show the first few rows of the dataset
print(data.head())

# Separate features and target variable
X = data.drop(columns=["Activity"])
y = data["Activity"]

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
deep_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
shallow_forest = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
deep_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train classifiers
shallow_tree.fit(X_train, y_train)
deep_tree.fit(X_train, y_train)
shallow_forest.fit(X_train, y_train)
deep_forest.fit(X_train, y_train)

# Make predictions
shallow_tree_pred = shallow_tree.predict(X_test)
deep_tree_pred = deep_tree.predict(X_test)
shallow_forest_pred = shallow_forest.predict(X_test)
deep_forest_pred = deep_forest.predict(X_test)

# Make probability predictions for log loss
shallow_tree_proba = shallow_tree.predict_proba(X_test)
deep_tree_proba = deep_tree.predict_proba(X_test)
shallow_forest_proba = shallow_forest.predict_proba(X_test)
deep_forest_proba = deep_forest.predict_proba(X_test)

# Calculate accuracies
shallow_tree_acc = accuracy_score(y_test, shallow_tree_pred)
deep_tree_acc = accuracy_score(y_test, deep_tree_pred)
shallow_forest_acc = accuracy_score(y_test, shallow_forest_pred)
deep_forest_acc = accuracy_score(y_test, deep_forest_pred)
# print("Accuracy:", shallow_tree_acc, deep_tree_acc, shallow_forest_acc, deep_forest_acc)

# Calculate precision
shallow_tree_prec = precision_score(y_test, shallow_tree_pred)
deep_tree_prec = precision_score(y_test, deep_tree_pred)
shallow_forest_prec = precision_score(y_test, shallow_forest_pred)
deep_forest_prec = precision_score(y_test, deep_tree_pred)
# print("Precision:", shallow_tree_prec, deep_tree_prec, shallow_forest_prec, deep_forest_prec)

# Calculate recall
shallow_tree_rec = recall_score(y_test, shallow_tree_pred)
deep_tree_rec = recall_score(y_test, deep_tree_pred)
shallow_forest_rec = precision_score(y_test, shallow_forest_pred)
deep_forest_rec = precision_score(y_test, deep_forest_pred)
# print("Recall:", shallow_tree_rec, deep_tree_rec, shallow_forest_rec, deep_forest_rec)

# Calculate F1-score
shallow_tree_f1 = f1_score(y_test, shallow_tree_pred)
deep_tree_f1 = f1_score(y_test, deep_tree_pred)
shallow_forest_f1 = f1_score(y_test, shallow_forest_pred)
deep_forest_f1 = f1_score(y_test, deep_forest_pred)
# print("F1-Score:", shallow_tree_f1, deep_tree_f1, shallow_forest_f1, deep_forest_f1)

# Calculate log loss
shallow_tree_logloss = log_loss(y_test, shallow_tree_proba)
deep_tree_logloss = log_loss(y_test, deep_tree_proba)
shallow_forest_logloss = log_loss(y_test, shallow_forest_proba)
deep_forest_logloss = log_loss(y_test, deep_forest_proba)
# print("Log Loss:", shallow_tree_logloss, deep_tree_logloss, shallow_forest_logloss, deep_forest_logloss)

# Print out the results of classifications
df_results = pd.DataFrame(columns=['Classifier Name', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Log Loss'])
df_results.loc[len(df_results.index)] = ['Shallow Tree', shallow_tree_acc, shallow_tree_prec, shallow_tree_rec,
                                         shallow_tree_f1, shallow_tree_logloss]
df_results.loc[len(df_results.index)] = ['Deep Tree', deep_tree_acc, deep_tree_prec, deep_tree_rec,
                                         deep_tree_f1, deep_tree_logloss]
df_results.loc[len(df_results.index)] = ['Shallow Forest', shallow_forest_acc, shallow_forest_prec, shallow_forest_rec,
                                         shallow_forest_f1, shallow_forest_logloss]
df_results.loc[len(df_results.index)] = ['Deep Forest', deep_forest_acc, deep_forest_prec, deep_forest_rec,
                                         deep_forest_f1, deep_forest_logloss]

print(df_results)

# Побудова Precision-Recall та ROC кривих для кожної моделі
models = {
    'Shallow Decision Tree': shallow_tree_proba[:, 1],
    'Deep Decision Tree': deep_tree_proba[:, 1],
    'Shallow Random Forest': shallow_forest_proba[:, 1],
    'Deep Random Forest': deep_forest_proba[:, 1]
}

plt.figure(figsize=(12, 5))

# Plot Precision-Recall curves
plt.subplot(1, 2, 1)
for model_name, y_proba in models.items():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label=f'{model_name} (AUC = {auc(recall, precision):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

# Plot ROC curves
plt.subplot(1, 2, 2)
for model_name, y_proba in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Лінія на графіку для випадкової моделі
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

thresholds_list = []
f1_scores = []

# Optimization task
# Train classifier
classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
recall = make_scorer(recall_score)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(classifier, param_grid, scoring=recall, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best recall: {grid_search.best_score_}")

y_pred = grid_search.best_estimator_.predict(X_test)
print('Classification report for task 5:\n',classification_report(y_test, y_pred))

# Choose the best threshold
y_proba_5 = grid_search.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 1.0, 0.01)
best_f1 = 0
threshold = 0

for t in thresholds:
    y_pred_optimal = np.where(y_proba_5 > t, 1, 0)
    f1 = f1_score(y_test, y_pred_optimal)

    thresholds_list.append(t)
    f1_scores.append(f1)

    if f1 > best_f1:
        best_f1 = f1
        threshold = t

print(f"The best threshold: {threshold}, F1-Score: {best_f1}")

thresholds_df = pd.DataFrame({
    'Threshold': thresholds_list,
    'F1-Score': f1_scores
})

print(thresholds_df)

y_pred_threshold = np.where(y_proba_5 > threshold, 1, 0)

print(f"Metrics for a classifier with  {threshold} threshold :")
print("Accuracy:", accuracy_score(y_test, y_pred_threshold))
print("Precision:", precision_score(y_test, y_pred_threshold))
print("Recall:", recall_score(y_test, y_pred_threshold))
print("F1-Score:", f1_score(y_test, y_pred_threshold))
print("Log Loss:", log_loss(y_test, grid_search.predict_proba(X_test)))
