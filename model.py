import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = {
    'age': [45, 34, 65, 50, 23, 70, 60, 40],
    'smoking': [1, 0, 1, 1, 0, 1, 1, 0],
    'yellow_fingers': [1, 0, 1, 1, 0, 1, 1, 0],
    'anxiety': [0, 0, 1, 1, 0, 1, 1, 0],
    'chest_pain': [1, 0, 1, 0, 0, 1, 1, 0],
    'coughing': [1, 0, 1, 1, 0, 1, 1, 0],
    'fatigue': [1, 1, 1, 1, 0, 1, 1, 0],
    'lung_cancer': [1, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)
X = df.drop('lung_cancer', axis=1)
y = df['lung_cancer']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("cancer_model.pkl", "wb"))

print("Model trained & saved!")