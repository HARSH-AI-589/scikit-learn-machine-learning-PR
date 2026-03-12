# Step 1: Import Libraries

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Step 2: Verify TensorFlow Installation

print("TensorFlow Version:", tf.__version__)
# Step 3: Create Logical Synthetic Dataset (1000 rows)

np.random.seed(42)

n = 1000

age = np.random.randint(22, 60, n)

experience = age - 22 + np.random.randint(-2, 5, n)
experience = np.clip(experience, 0, None)

salary = 30000 + (experience * 4000) + np.random.randint(-5000, 5000, n)

work_hours = np.random.randint(6, 11, n)

projects_completed = (experience // 2) + np.random.randint(0, 5, n)

performance_score = (
    (projects_completed * 2) +
    (work_hours * 1.5) +
    (experience * 0.5) +
    np.random.randint(-5, 5, n)
)

performance = (performance_score > np.median(performance_score)).astype(int)

data = pd.DataFrame({
    "Age": age,
    "Experience": experience,
    "Salary": salary,
    "Work_Hours": work_hours,
    "Projects_Completed": projects_completed,
    "Performance": performance
})

print("Logical dataset created successfully")
# Step 4: Check Dataset Shape and Preview

print("Dataset Shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())
# Step 5: Define Features (X) and Target (y)

X = data.drop("Performance", axis=1)
y = data["Performance"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
# Step 6: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)
print("\ny_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# Step 7: Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature scaling completed")
# Step 8: Build Neural Network Model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

print("\nNeural network model created")
# Step 9: Compile the Model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully")
# Step 10: Train the Model

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

print("\nModel training completed")
# Step 11: Evaluate Model on Test Data

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
# Step 12: Make Predictions

predictions = model.predict(X_test)

predicted_classes = (predictions > 0.5).astype(int)

print("\nFirst 10 Predictions:")
print(predicted_classes[:10])
# Step 13: Generate Final Exam Dataset (50,000 rows)

np.random.seed(100)

n_final = 50000

age_f = np.random.randint(22, 60, n_final)

experience_f = age_f - 22 + np.random.randint(-1, 3, n_final)
experience_f = np.clip(experience_f, 0, None)

salary_f = 30000 + (experience_f * 4200) + np.random.randint(-3000, 3000, n_final)

work_hours_f = np.random.randint(7, 11, n_final)

projects_completed_f = (experience_f // 2) + np.random.randint(1, 4, n_final)

performance_score_f = (
    (projects_completed_f * 2.5) +
    (work_hours_f * 1.7) +
    (experience_f * 0.6) +
    np.random.randint(-3, 3, n_final)
)

performance_f = (performance_score_f > np.median(performance_score_f)).astype(int)

final_data = pd.DataFrame({
    "Age": age_f,
    "Experience": experience_f,
    "Salary": salary_f,
    "Work_Hours": work_hours_f,
    "Projects_Completed": projects_completed_f,
    "Performance": performance_f
})

print("Final dataset shape:", final_data.shape)
# Step 14: Prepare Final Dataset

X_final = final_data.drop("Performance", axis=1)
y_final = final_data["Performance"]

X_final = scaler.transform(X_final)
# Step 15: Model Predictions

pred_final = model.predict(X_final)

pred_classes_final = (pred_final > 0.5).astype(int)
# Step 16: Calculate Final Accuracy

final_accuracy = (pred_classes_final.flatten() == y_final.values).mean() * 100

print("Final Test Accuracy:", final_accuracy)
# Step 17: Apply Cutoff Rule

cutoff = 92.88889999

if final_accuracy >= cutoff:
    print("MODEL SUCCESSFUL")
else:
    print("MODEL FAILED")