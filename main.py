import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def generateAI():
    # Load dataset
    dataset = pd.read_csv('data.csv')
    dataset = dataset.dropna()

    # Features (X) and labels (y)
    X = dataset.iloc[:, 1].values
    X = X.reshape(-1, 1)   # reshape correctly
    y = dataset.iloc[:, -1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    ai = KNeighborsClassifier(n_neighbors=5)
    ai.fit(X_train, y_train)

    # Evaluate model
    y_ai = ai.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_ai) * 100)

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(ai, f)
