# ---------------------------------------------
# Iris Flower Classification - Mohammed Sawood M
# CodeAlpha Data Science Internship | Task 1
# ---------------------------------------------

# 📌 Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 🔍 Step 1: Load the dataset
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target
df['species'] = df['species'].apply(lambda idx: iris_data.target_names[idx])

# 📊 Step 2: Explore the dataset
print("\n🔹 Preview of the dataset:\n")
print(df.head())

print("\n🔹 Dataset Information:\n")
print(df.info())

print("\n🔹 Species Count:\n")
print(df['species'].value_counts())

# 🎨 Step 3: Visualize feature relationships
sns.pairplot(df, hue='species')
plt.suptitle("Iris Flower Feature Pairplot", y=1.02)
plt.show()

# 🔥 Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 🧪 Step 4: Prepare data for training
X = df.drop('species', axis=1)
y = df['species']

# Encoding labels (Setosa, Versicolor, Virginica) to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ⚙️ Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 Step 6: Model Training using Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 📈 Step 7: Model Prediction and Evaluation
y_pred = model.predict(X_test_scaled)

# ✔️ Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# 🧮 Confusion Matrix
print("\n🔹 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 📋 Classification Report
print("\n🔹 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

