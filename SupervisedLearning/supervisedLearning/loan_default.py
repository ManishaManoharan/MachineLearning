import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("loan_approval_dataset.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Remove extra spaces from string values
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Drop loan_id if exists
if "loan_id" in data.columns:
    data = data.drop("loan_id", axis=1)

# Convert target column safely
data["loan_status"] = data["loan_status"].replace({
    "Approved": 1,
    "Rejected": 0
})

# Remove any rows where loan_status became NaN
data = data.dropna(subset=["loan_status"])

# Separate features and target
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# Convert categorical columns
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))