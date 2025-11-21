from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Sample dataset
df = pd.DataFrame({
    'Age': [45, 50, 55, 60, 65],
    'Cholesterol': [210, 230, 180, 240, 260],
    'HeartDisease': [0, 1, 0, 1, 1]
})

# Features and target
X = df[['Age', 'Cholesterol']]
y = df['HeartDisease']

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Train the model
pipe.fit(X, y)

# Predict for a new person
prediction = pipe.predict([[52, 200]])
print("Prediction:", prediction)
