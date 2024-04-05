import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

import cv2
import numpy as np

model = tf.keras.models.load_model('path_to_your_trained_model')

cap = cv2.VideoCapture(0)

label_map = {0: 'Without Mask', 1: 'With Mask'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess(frame)

    prediction = model.predict(processed_frame)
    label = label_map[int(prediction)]

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
