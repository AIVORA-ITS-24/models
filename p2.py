from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Sample dataset
df = pd.DataFrame({
    'Age': [20, 25, 30, 35, 40, 50],
    'Salary': [20000, 35000, 40000, 45000, 60000, 80000],
    'Purchased': [0, 1, 0, 1, 1, 1]
})

# Features & Target
X = df[['Age', 'Salary']]
y = df['Purchased']

# Create pipeline
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Train model
pipeline.fit(X, y)

# Predict
prediction = pipeline.predict([[28, 50000]])
print("Prediction:", prediction)

#
