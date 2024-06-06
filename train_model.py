import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import MobileNetV2
from preprocess import load_data, preprocess_data, split_data, augment_data

def build_model(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_data, val_data):
    model = build_model((224, 224, 3))
    datagen = augment_data()
    train_generator = datagen.flow_from_dataframe(train_data, directory='data/images/', x_col='filename', y_col='form_type', target_size=(224, 224), batch_size=32, class_mode='binary')
    val_generator = datagen.flow_from_dataframe(val_data, directory='data/images/', x_col='filename', y_col='form_type', target_size=(224, 224), batch_size=32, class_mode='binary')
    
    history = model.fit(train_generator, validation_data=val_generator, epochs=10)
    model.save('models/document_classifier.h5')
    return history

if __name__ == "__main__":
    data = load_data('data/sample_data.csv')
    data = preprocess_data(data)
    train, val = split_data(data)
    history = train_model(train, val)
    print("Model training complete.")
