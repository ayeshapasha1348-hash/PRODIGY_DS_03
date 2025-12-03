# task3.py
# ============================
# Task-03: Decision Tree Classifier - Bank Marketing Dataset
# Dark Background with Neon Lights (visuals using seaborn/matplotlib)
# ============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# 1) Load dataset
# ------------------------
# If you have downloaded the CSV file from GitHub or UCI
df = pd.read_csv("bank_dataset.csv")

print("✅ Dataset loaded successfully")
print(df.head())

# ------------------------
# 2) Check null values
# ------------------------
print("\nNull values in each column:")
print(df.isnull().sum())

# ------------------------
# 3) Handle missing/null values
# ------------------------
df['job'] = df['job'].fillna('unknown')

df['marital'].fillna('unknown', inplace=True)
df['education'].fillna('unknown', inplace=True)
df['balance'].fillna(0, inplace=True)
df['duration'].fillna(0, inplace=True)
df['pdays'].fillna(0, inplace=True)

print("\n✅ Null values after filling:")
print(df.isnull().sum())

# ------------------------
# 4) Prepare features and target
# ------------------------
# Target: y (yes=1, no=0)
y = df['y'].map({'yes':1, 'no':0})

# Features
X = df.drop('y', axis=1)

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
print("\nCategorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# ------------------------
# 5) Train/Test split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("\n✅ Train/Test split done")
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ------------------------
# 6) Build pipeline with Decision Tree
# ------------------------
preprocessor = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
], remainder='passthrough')

clf = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'))
])

print("\nTraining the Decision Tree...")
clf.fit(X_train, y_train)
print("✅ Training complete")

# ------------------------
# 7) Predictions & Evaluation
# ------------------------
y_pred = clf.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix (Dark background + neon style)
plt.style.use('dark_background')
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='cool', xticklabels=['No','Yes'], yticklabels=['No','Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ------------------------
# 8) Feature Importance
# ------------------------
ohe = clf.named_steps['pre'].named_transformers_['ohe']
feature_names = list(ohe.get_feature_names_out(cat_cols)) + num_cols
importances = clf.named_steps['model'].feature_importances_

fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi.sort_values(by='importance', ascending=False, inplace=True)
print("\nTop 10 Feature Importances:\n", fi.head(10))

# Plot Feature Importance
plt.figure(figsize=(8,6))
sns.barplot(data=fi.head(10), x='importance', y='feature', palette='cool')
plt.title("Top 10 Feature Importances")
plt.show()

# ------------------------
# 9) Optional: Plot Decision Tree (top 3 levels)
# ------------------------
plt.figure(figsize=(20,10))
plot_tree(clf.named_steps['model'],
          feature_names=feature_names,
          class_names=['No','Yes'],
          filled=True,
          rounded=True,
          max_depth=3,
          fontsize=10)
plt.title("Decision Tree (Top 3 Levels)")
plt.show()

print("\n✅ Task-03 Complete!")

# ------------------------
# Save cleaned dataset
# ------------------------
df.to_csv("bank_dataset_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as CSV: bank_dataset_cleaned.csv")

