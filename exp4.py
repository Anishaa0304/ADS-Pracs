from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import math
import matplotlib.pyplot as plt

# Load Dataset
dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Training
clf_tree = DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)

# Predictions
y_pred = clf_tree.predict(X_test)
print("Predictions:", y_pred)

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"True Negatives: {tn}\nFalse Negatives: {fn}\nTrue Positives: {tp}\nFalse Positives: {fp}")

# Evaluation Metrics
acc = (tn + tp) / (tn + tp + fn + fp)
error_rate = (fn + fp) / (tn + tp + fn + fp)
precision = tp / (tp + fp)
sns = tp / (tp + fn)
spc = tn / (tn + fp)
roc = math.sqrt((sns**2 + spc**2) / 2)
GM = math.sqrt(sns * spc)
f1 = (2 * sns * precision) / (precision + sns)
fpr = 1 - spc
fnr = 1 - sns
power = 1 - fnr

print(f"Accuracy: {acc}")
print(f"Error Rate: {error_rate}")
print(f"Precision: {precision}")
print(f"Sensitivity: {sns}")
print(f"Specificity: {spc}")
print(f"ROC: {roc}")
print(f"Geometric Mean: {GM}")
print(f"F1 Score: {f1}")
print(f"False Positive Rate: {fpr}")
print(f"False Negative Rate: {fnr}")
print(f"Power: {power}")

# ROC Curve
false_positive_rate1, true_positive_rate1, _ = roc_curve(y_test, y_pred)
print('roc_auc_score for DecisionTree:', roc_auc_score(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate1, true_positive_rate1, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend()
plt.grid(True)
plt.show()
