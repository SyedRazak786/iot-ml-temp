import pandas as pandas

def generateAI()
dataset=pd.read_csv('data.csv')
dataset=dataset.dropna()
X=dataset.iloc[:,1].values
X=X.reshape(-1.1)
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
ai=KNeighborsClassifier(n_neighbors = 5)

ai.fit(X_train,y_train)
ai.fit(X_train,y_train)
ai.predict([[25]])
y_ai=ai.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_ai)*100)
import pickle 

pickle.dump(ai,open('model.pkl','wb'))