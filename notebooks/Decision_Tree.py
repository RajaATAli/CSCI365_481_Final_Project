import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the dataset (adjust path as needed)
data_one_hot = pd.read_csv('../data/processed/cleaned_diabetes_one_hot_encoding.csv')

# Splitting the data into features and target
X = data_one_hot.drop(columns=['diabetes'])
y = data_one_hot['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment with different max_depth values
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    decision_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    decision_tree.fit(X_train, y_train)
    train_scores.append(decision_tree.score(X_train, y_train))
    test_scores.append(decision_tree.score(X_test, y_test))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, marker='o', label='Training Accuracy')
plt.plot(depths, test_scores, marker='o', label='Testing Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Effect of Max Depth on Decision Tree Performance')
plt.legend()
plt.grid()
plt.show()

# Experiment with different min_samples_split values
splits = range(2, 21)
train_scores = []
test_scores = []

for split in splits:
    decision_tree = DecisionTreeClassifier(min_samples_split=split, random_state=42)
    decision_tree.fit(X_train, y_train)
    train_scores.append(decision_tree.score(X_train, y_train))
    test_scores.append(decision_tree.score(X_test, y_test))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(splits, train_scores, marker='o', label='Training Accuracy')
plt.plot(splits, test_scores, marker='o', label='Testing Accuracy')
plt.xlabel('Min Samples Split')
plt.ylabel('Accuracy')
plt.title('Effect of Min Samples Split on Decision Tree Performance')
plt.legend()
plt.grid()
plt.show()

# Experiment with different min_samples_leaf values
leaves = range(1, 21)
train_scores = []
test_scores = []

for leaf in leaves:
    decision_tree = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
    decision_tree.fit(X_train, y_train)
    train_scores.append(decision_tree.score(X_train, y_train))
    test_scores.append(decision_tree.score(X_test, y_test))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(leaves, train_scores, marker='o', label='Training Accuracy')
plt.plot(leaves, test_scores, marker='o', label='Testing Accuracy')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Accuracy')
plt.title('Effect of Min Samples Leaf on Decision Tree Performance')
plt.legend()
plt.grid()
plt.show()

# Final model with selected parameters
decision_tree = DecisionTreeClassifier(
    max_depth=5,  # You can change based on your experiments
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
decision_tree.fit(X_train, y_train)

# Making predictions and calculating probabilities for ROC-AUC
y_pred = decision_tree.predict(X_test)
y_proba = decision_tree.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)  # Calculating the ROC-AUC score

# Displaying the results
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)
print('Confusion Matrix:')
print(conf_matrix)
print(f'ROC-AUC Score: {roc_auc}')

# Plotting the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {roc_auc:.2f})", marker='.')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Visualizing the final Decision Tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.show()
