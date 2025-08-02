import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Load and train the MNIST model
# ------------------------------
@st.cache_resource
def load_and_train_model():
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), verbose=0)

    return model, (x_test, y_test)

# ------------------------------
# Main Streamlit App
# ------------------------------
def main():
    st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
    st.title("🎯 MNIST Digit Recognizer (Streamlit + CNN)")

    # Load model and test data
    with st.spinner("Training the CNN model... (only once, cached)"):
        model, (x_test, y_test) = load_and_train_model()
    st.success("✅ Model trained and ready!")

    # Select image index
    index = st.slider("🔢 Pick a test image index (0 to 9999):", 0, 9999, 0)

    # Prepare image and label
    image = x_test[index].reshape(28, 28)
    true_label = y_test[index]

    # Make prediction
    prediction = model.predict(np.expand_dims(x_test[index], axis=0), verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Show image and prediction
    st.image(image, caption=f"Actual: {true_label}, Predicted: {predicted_label}", width=200, channels="L")
    st.write(f"🧠 The model is **{confidence:.2f}%** confident that this is a **'{predicted_label}'**.")

    # Show prediction probabilities
    st.subheader("📊 Prediction Probabilities")
    probabilities = prediction[0]
    fig, ax = plt.subplots()
    ax.bar(range(10), probabilities, color='skyblue')
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
