import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
# Define data augmentation parameters
import seasonXXlooder
import seasonYYlooder

df_alg,df_nonalg,df_nonveg=seasonYYlooder.load()
#df_alg0,df_nonalg0,df_nonveg0=seasonXXlooder.load()
#df_alg1,df_nonalg1,df_nonveg1=seasonYYlooder.load()
#df_alg=pd.concat([df_alg0,df_alg1],ignore_index=True)
#df_nonalg=pd.concat([df_nonalg0,df_nonalg1],ignore_index=True)
#df_nonveg=pd.concat([df_nonveg0,df_nonveg1],ignore_index=True)

xtrain= pd.concat([df_alg.iloc[:,45:-1],df_nonalg.iloc[:,45:-1], df_nonveg.iloc[:,45:-1]],ignore_index=True)
#xtrain= pd.concat([df_alg.iloc[:,2:7],df_nonalg.iloc[:,2:7], df_nonveg.iloc[:,2:7]],ignore_index=True)
#scaler = MinMaxScaler()
#xtrain= pd.DataFrame(scaler.fit_transform(xtrain), columns=xtrain.columns)
ytrain= pd.concat([df_alg.iloc[:,-1:],df_nonalg.iloc[:,-1:], df_nonveg.iloc[:,-1:]],ignore_index=True)




data= pd.concat([df_alg.iloc[:,45:],df_nonalg.iloc[:,45:], df_nonveg.iloc[:,45:]],ignore_index=True)
print(data.shape)
# Assuming the last column is the target label
X = data.iloc[:,:-1].values  # Features
#print(X.shape)
y = data.iloc[:, -1].values   # Labels

# Reshape the features
X_reshaped = X.reshape(-1, 3, 3, 10)  # Reshape to 3D array with 5 channels
print("1")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
# class_0_generator = datagen.flow(X_train[y_train == 0], y_train[y_train == 0], batch_size=32)
# print("2")

# # Create a new generator for class 1
# class_1_generator = datagen.flow(X_train[y_train == 1], y_train[y_train == 1], batch_size=32)
# print("2_")
# combined_generator = np.concatenate((class_0_generator, class_1_generator), axis=0)
# print("2__")
# class_2_data = X_train[y_train == 2]
# class_2_labels = y_train[y_train == 2]
# print("3")
# # Combine the original data for class 3 with the augmented data for classes 0 and 1
# combined_data = np.concatenate((combined_generator, class_2_data), axis=0)
# combined_labels = np.concatenate((np.zeros(len(class_0_generator)), np.ones(len(class_1_generator)), class_2_labels))
# print("4")
# # Shuffle the combined data
# combined_indices = np.random.permutation(len(combined_data))
# combined_data = combined_data[combined_indices]
# combined_labels = combined_labels[combined_indices]

# print(len(X_train))
# print(len(combined_data))
input_shape = (3, 3, 10)
# Define the CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, kernel_size=(1, 1), activation='leaky_relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(1, 1), activation='leaky_relu'))

# Flatten the output before feeding into dense layers
model.add(Flatten())

# Add dense layers
model.add(Dense(64, activation='leaky_relu'))
model.add(Dense(3, activation='softmax'))  # Output size is 3, using softmax activation for multi-class classification

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint_callback = ModelCheckpoint(filepath='cnn/270424/modelY_epoch_{epoch:02d}.h5', save_freq='epoch', save_weights_only=False, verbose=0)


# Print model summary
model.summary()
with open('cnn/270424/model_summary.txt', 'w') as f:
    # Redirect print output to the text file
    model.summary(print_fn=lambda x: f.write(x + '\n'))
output_file_path = 'cnn/270424/training_logs.txt'
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            f.write(f"Epoch {epoch+1}/{self.params['epochs']}: loss={logs['loss']}, accuracy={logs['accuracy']}, val_loss={logs['val_loss']}, val_accuracy={logs['val_accuracy']}\n")

# Create custom callback instance
custom_callback = CustomCallback(output_file_path)

# Train the model
history=model.fit(X_train,y_train, epochs=200, batch_size=64, validation_split=0.2,callbacks=[checkpoint_callback, custom_callback])


# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

ypred= np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, ypred)
print(cm)
precision = precision_score(y_test, ypred, average='weighted')
recall = recall_score(y_test, ypred, average='weighted')
f1 = f1_score(y_test, ypred, average='weighted')
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print('Test accuracy:', test_acc)
with open('history.txt', 'w') as f:
    f.write(str(history.history))

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('cnn/2704032024/loss_plot.png')
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('cnn/27042024/accuracy_plot.png')
plt.show()