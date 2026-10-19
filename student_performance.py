#-----------------------Student Preformance Prediction-----------------#
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv('student-mat.csv', sep=";")

print("--- Dataset Head ---")
print(data.head())
print("\n--- Dataset Description ---")
print(data.describe())
print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\nMean study time:", data['studytime'].mean())
print("Mean failures:", data['failures'].mean())
print("Mean G1:", data['G1'].mean())
print("Mean G2:", data['G2'].mean())
print("Mean G3:", data['G3'].mean())
print("Mean absences:", data['absences'].mean())

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
sns.scatterplot(x=data['G1'], y=data['G2'])
plt.xlabel("G1")
plt.ylabel("G2")

plt.subplot(2, 3, 2)
sns.scatterplot(x=data['G2'], y=data['G3'])
plt.xlabel("G2")
plt.ylabel("G3")

plt.subplot(2, 3, 3)
sns.scatterplot(x=data['absences'], y=data['G3'])
plt.xlabel("Absences")
plt.ylabel("G3")

plt.subplot(2, 3, 4)
sns.scatterplot(x=data['failures'], y=data['G3'])
plt.xlabel("Failures")
plt.ylabel("G3")

plt.subplot(2, 3, 5)
sns.scatterplot(x=data['studytime'], y=data['G3'])
plt.xlabel("Study Time")
plt.ylabel("G3")

plt.tight_layout()
plt.show()


data_cleaned = data.copy()
Q1 = np.percentile(data_cleaned['absences'], 25)
Q3 = np.percentile(data_cleaned['absences'], 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = [x for x in data_cleaned['absences'] if x < lower or x > upper]
print(f"Number of absence outliers: {len(outliers)}")

X = data_cleaned[['G1', 'G2', 'studytime', 'absences']]
y = data_cleaned['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
lr_pred = model_lr.predict(X_test)

print("\n--- Linear Regression Performance ---")
print("R2 Score:", r2_score(y_test, lr_pred))
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])

params = {
    'knn__n_neighbors': [3, 5, 7, 9]
}

grid = GridSearchCV(
    knn_pipeline,
    params,
    cv=5,
    scoring='r2'
)

grid.fit(X_train, y_train)

print("\n--- Tuned KNN Performance ---")
print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

knn_pred = grid.predict(X_test)
print("KNN R2:", r2_score(y_test, knn_pred))
print("KNN MAE:", mean_absolute_error(y_test, knn_pred))
print("KNN MSE:", mean_squared_error(y_test, knn_pred))

results = {
    'Model': ['Linear Regression', 'Tuned KNN'],
    'R2 Score': [r2_score(y_test, lr_pred), r2_score(y_test, knn_pred)],
    'MAE': [mean_absolute_error(y_test, lr_pred), mean_absolute_error(y_test, knn_pred)],
    'MSE': [mean_squared_error(y_test, lr_pred), mean_squared_error(y_test, knn_pred)]
}

results_df = pd.DataFrame(results)
print("\n--- Model Comparison Summary ---")
print(results_df)