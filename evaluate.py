#workload recommended

import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import numpy as np
import os
import time

MODEL_FOLDER = "models/"

# List of models to evaluate

model_files = [
    "original_model.h5",
    "quantized_model.tflite",
    "int8_quantized_model.tflite",
    "float16_quantized_model.tflite"
]


# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
test_images = test_images / 255.0

# evaluate a tflite model

def evaluate_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    input_type = input_details[0]['dtype']

    correct = 0
    total = len(test_images)

    start = time.time()

    for i in range(total):
        img = test_images[i].reshape(1, 28, 28)

        #match model input type
        if input_type == np.float32:
            input_data = img.astype(np.float32)
        elif input_type == np.uint8:
            input_data = (img * 255).astype(np.uint8)
        elif input_type == np.int8:
            input_data = ((img - 0.5) * 255).astype(np.int8)
        else:
            raise ValueError("unsupported input type:", input_type)

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output)

        if predicted == test_labels[i]:
            correct += 1
    
    accuracy = correct / total
    latency = time.time() - start

    return accuracy, latency

print("\nEvaluating all models:\n")
print(f"{'Model':<25} {'Size (MB)':<25} {'Accuracy {%}':<15} {'Time (s)':<10}")
print("-" * 60)

for mf in model_files:
    path = os.path.join(MODEL_FOLDER, mf)
    size_mb = os.path.getsize(path) / (1024 * 1024)

    if mf.endswith(".h5"): # Keras model
        model = tf.keras.models.load_model(path)
        loss, acc = model.evaluate(test_images, test_labels, verbose = 0)
        latency = 0.0 # not measured
    else: #TFlite model
        acc, latency = evaluate_tflite(path)

    print(f"{mf:<25} {size_mb:<25} {acc*100:<15.2f} {latency:<10.2f}")


print("\nFinished evaluation of all models.")






