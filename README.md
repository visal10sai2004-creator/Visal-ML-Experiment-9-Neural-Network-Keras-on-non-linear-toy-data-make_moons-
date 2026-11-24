## Experiment-9.Neural Network (Keras) on non-linear toy data (make_moons)

## AIM:

Train a small feedforward neural network to classify non-linear data.

## ALGORITHM:

Two-layer dense network with ReLU hidden units and sigmoid output, trained with Adam.

## ALGORITHM:

Two-layer dense network with ReLU hidden units and sigmoid output, trained with Adam.

## PROGRAM:

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X,y = make_moons(n_samples=500, noise=0.2, random_state=42)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2, random_state=42)
model = Sequential([Dense(8, activation='relu', input_shape=(2,)), 
Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])
model.fit(Xtr, ytr, epochs=30, batch_size=16, verbose=0)
y_pred = (model.predict(Xte) > 0.5).astype(int).flatten()
print('NN test accuracy:', round(accuracy_score(yte, y_pred),4))

```

## OUTPUT:

```

NN test accuracy: 0.86

```

## RESULT:

Neural networks model non-linear boundaries; students see improved accuracy over linear methods
