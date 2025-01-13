import os   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This suppresses most TensorFlow logs.
import numpy as np
import matplotlib.pyplot as plt                       
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters


#base_dir = "C:/Users/Joshuaa/OneDrive/Documents/Study/ODL - Mr. Raheem Mafas/Group Assignment/image_dataset"  # Change this to the actual path of the extracted dataset
base_dir = "C:/Users/Ikraam02/OneDrive - Asia Pacific University/Documents/Year 3 Semester 2/archive"
train_dir = os.path.join(base_dir, 'seg_train')
test_dir = os.path.join(base_dir, 'seg_test')

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=64,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=64,
    class_mode='categorical',
    subset='validation')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(100, 100),
                                                  batch_size=64,
                                                  class_mode='categorical')


# Define the model building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(100, 100, 3)))
    
    # Tune the number of units in the first Dense layer
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))
    
    # Tune the number of units in the second Dense layer
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))
    
    # Tune the number of units in the third Dense layer
    model.add(Dense(units=hp.Int('units_3', min_value=32, max_value=256, step=16), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_3', 0.2, 0.5, step=0.1)))
    
    model.add(Dense(6, activation='softmax'))
    
    # Tune the learning rate for the Adam optimizer
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 5e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Define the Keras Tuner RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Number of models to try
    executions_per_trial=1,  # Number of different executions for each model
    directory='hyperparameter_tuning',
    project_name='image_classification_tuning')

# Search for the best hyperparameters
tuner.search_space_summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Perform the search
tuner.search(train_generator,
             epochs=10,
             validation_data=validation_generator,
             callbacks=[early_stopping])

# Get the best model and hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

model.save('best_ann_model.keras')

# Train the best model
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
Y_true = test_generator.classes
test_generator.reset()
Y_pred = np.argmax(model.predict(test_generator), axis=1)

# Classification report
print("Classification Report:\n", classification_report(Y_true, Y_pred, target_names=test_generator.class_indices.keys()))

# Confusion matrix
conf_matrix = confusion_matrix(Y_true, Y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_generator.class_indices.keys(), 
            yticklabels=test_generator.class_indices.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
