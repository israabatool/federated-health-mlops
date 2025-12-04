# src/tff_train.py
import os
import pickle
import tensorflow as tf
import tensorflow_federated as tff

# Old import
# from models.create_model import create_keras_model

# Correct import when running as module
from src.models.create_model import create_keras_model

from federated_training.tff_client_data import load_client_data

# -------------------------------
# Setup paths and output folder
# -------------------------------
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load federated data (Phase 3)
# -------------------------------
client_datasets = load_client_data()  # Dictionary of {client_name: tf.data.Dataset}
client_names = list(client_datasets.keys())
print(f"Loaded clients: {client_names}")

# Convert dictionary to list of datasets for TFF
federated_train_data = [client_datasets[name] for name in client_names]

# -------------------------------
# Define TFF model function
# -------------------------------
def model_fn():
    keras_model = create_keras_model()  # Your Keras model architecture
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# Build Federated Averaging process
fed_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

# -------------------------------
# Initialize server state
# -------------------------------
state = fed_process.initialize()

# -------------------------------
# Federated Training Loop
# -------------------------------
NUM_ROUNDS = 5  # Increase as needed
for round_num in range(1, NUM_ROUNDS + 1):
    result = fed_process.next(state, federated_train_data)
    state = result.state
    train_metrics = result.metrics['client_work']['train']
    print(f"Round {round_num} metrics: "
          f"accuracy={train_metrics['binary_accuracy']:.4f}, "
          f"loss={train_metrics['loss']:.4f}, "
          f"num_examples={train_metrics['num_examples']}")

# -------------------------------
# Save Final Federated Model
# -------------------------------
keras_model = create_keras_model()
fed_process.get_model_weights(state).assign_weights_to(keras_model)

# Save full Keras model
keras_model.save(os.path.join(OUTPUT_DIR, 'final_federated_model.h5'))

# Save weights separately with pickle
with open(os.path.join(OUTPUT_DIR, 'final_fed_model_weights.pkl'), 'wb') as f:
    pickle.dump(fed_process.get_model_weights(state), f)

print(f"Federated model saved successfully in '{OUTPUT_DIR}/'")

