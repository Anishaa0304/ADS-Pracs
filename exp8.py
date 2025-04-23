from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from numpy import where
from collections import Counter
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv(r"C:\Users\anish\Downloads\ADS lab exam solutions\ADS Datasets\Churn_Modelling.csv")

# Initial scatter plot
data = df[['CreditScore', 'Age', 'Exited']]
sns.scatterplot(data=data, x='CreditScore', y='Age', hue='Exited')
plt.title('Initial CreditScore vs Age Scatter Plot')
plt.show()

# Encode categorical features
for col in df.columns:
    if df[col].dtype == 'O':
        label_encode = LabelEncoder()
        df[col] = label_encode.fit_transform(df[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Exited', axis=1), df['Exited'], test_size=0.2, random_state=101)

# Decision Tree before SMOTE
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Decision Tree (Before SMOTE):")
print(classification_report(y_test, y_pred))

# Decision Tree after SMOTE
smote = SMOTE(random_state=101)
X_oversample, y_oversample = smote.fit_resample(X_train, y_train)
clf.fit(X_oversample, y_oversample)
y_predo = clf.predict(X_test)
print("Decision Tree (After SMOTE):")
print(classification_report(y_test, y_predo))

# Logistic Regression before SMOTE
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
print("Logistic Regression (Before SMOTE):")
print(classification_report(y_test, classifier.predict(X_test)))

# Logistic Regression after SMOTE
classifier.fit(X_oversample, y_oversample)
print("Logistic Regression (After SMOTE):")
print(classification_report(y_test, classifier.predict(X_test)))

# Visualizing SMOTE oversampling on two features
X, y = smote.fit_resample(df[['CreditScore', 'Age']], df['Exited'])
df_oversampler = pd.DataFrame(X, columns=['CreditScore', 'Age'])
df_oversampler['Exited'] = y

# Count plot after oversampling
sns.countplot(data=df_oversampler, x='Exited')
plt.title('Class Distribution After SMOTE')
plt.show()

# Final scatter plot after SMOTE
sns.scatterplot(data=df_oversampler, x='CreditScore', y='Age', hue='Exited')
plt.title('CreditScore vs Age After SMOTE')
plt.show()

# Display new class balance
counter = Counter(y)
print("Class Distribution After SMOTE:", counter)
