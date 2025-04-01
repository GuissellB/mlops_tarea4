
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# --------------------------------------------
# READ THE DATASET
# --------------------------------------------
df = pd.read_csv("diabetes.csv")

# --------------------------------------------
# PREPROCESSING
# --------------------------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------
# PIPELINE
# --------------------------------------------
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# --------------------------------------------
# HYPERPARAMETER OPTIMIZATION
# --------------------------------------------
param_grid_logreg = {
    'classifier__C': [0.1, 1, 10, 100]
}

# --------------------------------------------
# TRAINING
# --------------------------------------------
grid_search_log_reg = GridSearchCV(pipeline, param_grid_logreg, cv=5, n_jobs=-1)
grid_search_log_reg.fit(X_train, y_train)

# --------------------------------------------
# BEST MODEL SELECTION
# --------------------------------------------
best_model = grid_search_log_reg.best_estimator_

# --------------------------------------------
# METRICS
# --------------------------------------------
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

with open("metrics_diabetes.txt", 'w') as outfile:
    outfile.write("Classification Report:\n")
    outfile.write(report)
    outfile.write("\nConfusion Matrix:\n")
    outfile.write(str(conf_matrix))

# ROC Curve
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC Curve (area = {:.2f})".format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Diabetes Prediction")
plt.legend(loc="lower right")
plt.savefig("roc_diabetes.png", dpi=120)
plt.close()

# --------------------------------------------
# SERIALIZING
# --------------------------------------------
model_filename = "model/diabetes_logistic_model.pkl"
joblib.dump(best_model, model_filename)
