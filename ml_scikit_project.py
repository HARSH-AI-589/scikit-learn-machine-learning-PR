# Step 1 — Import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score


# Step 2 — Create a Logical Dataset (500 rows)

np.random.seed(42)

age = np.random.randint(18, 60, 500)
income = np.random.randint(20000, 100000, 500)
spending_score = np.random.randint(1, 100, 500)

# Logical rule for purchase
purchased = ((income > 50000) & (spending_score > 50)).astype(int)

data = {
    "Age": age,
    "Income": income,
    "SpendingScore": spending_score,
    "Purchased": purchased
}

df = pd.DataFrame(data)

print(df.head())
print("Dataset Shape:", df.shape)


# Step 3 — Separate Features and Target

X = df[['Income', 'SpendingScore']]
y = df['Purchased']

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# Step 4 — Split Dataset (80% Training, 20% Testing)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# Step 5 — Create the Model

model = LogisticRegression()


# Step 6 — Train the Model

model.fit(X_train, y_train)

print("\nModel training completed. The model is now trained.")


# Step 7 — Make Predictions

y_pred = model.predict(X_test)

print("\nPredictions (first 10):")
print(y_pred[:10])


# Step 8 — Check Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nPrecision:", precision)
print("Recall:", recall)
# Step 17 — Create new logical dataset (50 rows)

np.random.seed(100)

new_income = np.random.randint(20000, 100000, 50)
new_spending_score = np.random.randint(1, 100, 50)

# same logical rule used earlier
new_purchased = ((new_income > 50000) & (new_spending_score > 50)).astype(int)

new_data = pd.DataFrame({
    "Income": new_income,
    "SpendingScore": new_spending_score,
    "ActualPurchased": new_purchased
})

print("\nNew Dataset (50 customers):")
print(new_data)
# Step 18 — Model predictions on new dataset

X_new = new_data[['Income', 'SpendingScore']]

new_predictions = model.predict(X_new)

new_data["PredictedPurchased"] = new_predictions

print("\nModel Predictions on New Dataset:")
print(new_data)
new_accuracy = accuracy_score(new_data["ActualPurchased"], new_data["PredictedPurchased"])

print("\nAccuracy on new dataset:", new_accuracy)