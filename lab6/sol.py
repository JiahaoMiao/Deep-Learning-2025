import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# load iris
iris = load_iris()

# pandas
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['label'] = iris.target_names[iris.target]

print(df)
sns.pairplot(df, hue='label')
plt.show()

# dataset
train_dataset = df.sample(frac=0.8, random_state=1)
test_dataset = df.drop(train_dataset.index)

# one hot encoding
label = pd.get_dummies(train_dataset['label'], prefix='label')
df = pd.concat([df, label], axis=1)
df = df.drop(['label'])  # Drop 'label' column after encoding
print(df)

x_train = train_dataset[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y_train = df[['label_setosa', 'label_versicolor', 'label_virginica']]

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train,
    y_train, 
    epochs=200, 
    validation_split=0.4,  # 40% of the data will be used for validation
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        tf.keras.callbacks.TensorBoard(log_dir='./logs/fit')
    ]
)

# plot loss
train_metrics = history.history['loss']
val_metrics = history.history['val_loss']
epochs = range(1, len(train_metrics) + 1)

plt.plot(epochs, train_metrics, 'bo', label='Training loss')
plt.plot(epochs, val_metrics, 'b', label='Validation loss')
plt.grid()
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()
