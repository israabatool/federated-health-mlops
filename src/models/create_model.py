import tensorflow as tf

def create_keras_model():
    """
    Create a simple binary classification Keras model.
    IMPORTANT: Do NOT compile the model for TFF.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),  # 5 features: age, gender, bp, glucose, hr
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model  

