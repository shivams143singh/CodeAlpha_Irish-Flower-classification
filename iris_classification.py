# ==========================================
# STEP 1: Import the Required Libraries
# ==========================================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# STEP 2: Load the Iris Dataset
# ==========================================
# Scikit-learn has the Iris dataset built-in for easy access
iris = load_iris()

# X represents the input features (4 measurements)
X = iris.data 
# y represents the target labels (0, 1, or 2 representing the species)
y = iris.target 

print("--- Dataset Summary ---")
print(f"Feature Names: {iris.feature_names}")
print(f"Target Species: {iris.target_names}")
print(f"Total rows of data: {X.shape[0]}\n")

# ==========================================
# STEP 3: Split Data into Training and Testing sets
# ==========================================
# We reserve 20% of the data to evaluate the model's true performance later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4: Train the Machine Learning Model
# ==========================================
# We use a Decision Tree Classifier, which learns human-like if/then rules to classify flowers
model = DecisionTreeClassifier(random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)
print("Model training completed successfully.\n")

# ==========================================
# STEP 5: Evaluate the Model's Performance
# ==========================================
# Use the trained model to make predictions on the hidden test data
y_pred = model.predict(X_test)

# Calculate the overall accuracy score
# Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
accuracy = accuracy_score(y_test, y_pred)

print("--- Evaluation Results ---")
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%\n")

# Print a detailed report showing precision, recall, and f1-score for each species
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
