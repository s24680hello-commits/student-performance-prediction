Student Performance Prediction using Machine Learning

📌 Project Overview

This project predicts students' final mathematics grades using machine learning. It analyzes factors such as previous grades, study time, and absences to estimate a student's final grade.

The project demonstrates a basic machine learning workflow, including Exploratory Data Analysis (EDA), data preprocessing, model training, hyperparameter tuning, and model evaluation.

🎯 Objectives

- Explore and visualize student performance data.
- Identify patterns and relationships between student attributes and final grades.
- Train regression models to predict final mathematics grades.
- Tune K-Nearest Neighbors (KNN) using GridSearchCV.
- Compare model performance using evaluation metrics.

📂 Dataset

Dataset: Student Performance — Mathematics ("student-mat.csv")

Original source: "UCI Machine Learning Repository — Student Performance" (https://archive.ics.uci.edu/dataset/320/student+performance)

Original creator: Paulo Cortez

License: "Creative Commons Attribution 4.0 International (CC BY 4.0)" (https://creativecommons.org/licenses/by/4.0/)

The dataset contains information about students, including their previous grades, study time, absences, and final mathematics grades.

The original dataset uses semicolons (";") as separators.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

🔍 Project Workflow

1. Data Loading and Exploration

- Loaded the dataset using Pandas.
- Examined the dataset's structure, columns, and data types.
- Checked for missing values and duplicate records.

2. Exploratory Data Analysis (EDA)

- Created scatter plots to explore relationships between variables.
- Used boxplots to examine the distribution of data and potential outliers.
- Examined the relationship between previous grades and final grades.

3. Data Preprocessing

- Selected relevant input features.
- Separated the features ("X") from the target ("y").
- Applied one-hot encoding where required.
- Split the data into training and testing sets.
- Used feature scaling for the KNN model.

4. Model Training

Two regression models were trained:

- Linear Regression: Used to model relationships between input features and final grades.
- K-Nearest Neighbors (KNN) Regression: Used to predict grades based on similar data points.

5. Hyperparameter Tuning

GridSearchCV was used to find a suitable value of "n_neighbors" for the KNN model, with 5-fold cross-validation.

The best parameter found was:

"n_neighbors = 7"

6. Model Evaluation

The models were evaluated using:

- R² Score: Measures how well the model explains variation in the target.
- Mean Absolute Error (MAE): Measures the average absolute prediction error.
- Mean Squared Error (MSE): Measures the average squared prediction error.

📊 Results

The following results were obtained from the test set during the project run.

Evaluation Metric| Linear Regression| Tuned KNN Regression
R² Score| 0.7961| 0.8158
MAE| 1.3126| 1.3092
MSE| 4.1790| 3.7763

Best KNN parameter: "n_neighbors = 7"

In this run, tuned KNN performed slightly better than Linear Regression across the three reported test metrics. Results may vary if the dataset, selected features, or train-test split changes.

🚀 How to Run the Project

Prerequisites

Install Python 3 and Git (Git is needed only if you want to clone the repository).

1. Clone the repository

git clone https://github.com/s24680hello-commits/student-performance-prediction.git

Replace "YOUR-USERNAME" with your GitHub username.

2. Open the project folder

cd student-performance-prediction

3. Install the required libraries

pip install -r requirements.txt

4. Run the Python file

python student_performance.py

Make sure "student_performance.py" matches the actual name of your Python file and that "student-mat.csv" is in the same folder.

📁 Project Structure

student-performance-prediction/
│
├── student_performance.py
├── student-mat.csv
├── requirements.txt
└── README.md

⚠️ Limitations

- The model uses only selected features from the dataset.
- Predictions are estimates and may not reflect a student's actual final grade.
- The dataset represents a particular group of students and may not generalize to every school or education system.
- Previous grades are strong predictors of final grades, so the model's performance should be interpreted in the context of the available features.

🔮 Future Improvements

- Experiment with additional regression algorithms.
- Compare feature combinations and their effect on model performance.
- Improve the model evaluation process.
- Build a simple web application to demonstrate predictions.

👨‍💻 Author

Created as a student machine learning project to practice data analysis, regression, model evaluation, and hyperparameter tuning.

📜 Dataset Attribution

Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. Please refer to the original "UCI dataset page" (https://archive.ics.uci.edu/dataset/320/student+performance) for dataset details and attribution information.