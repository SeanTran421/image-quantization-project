#main.py done.

# import tensor flow and load dependencies
import tensorflow as tf
from tensorflow.keras import datasets, models, layers

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# normalize the data. divide by 255 scales the values to the range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

#Build a simple neural network model

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# Compile the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model

model.fit(train_images, train_labels, epochs=5)

# After training the model, save the model for further tests
model.save("models/original_model.h5")

#Convert the model to a quanitzed version (dynamic range)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Save the quantized model to a .tflite file

with open('models/quantized_model.tflite', 'wb') as f:
    f.write(quantized_model)
print("Dynamic range quantized model saved as 'quantized_model.tflite'")

# Next, INT8 Quantization (Full Integer)

def representative_dataset():
    for i in range(100):
        yield [train_images[i].reshape(1, 28, 28).astype('float32')]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_int8 = converter.convert()

with open('models/int8_quantized_model.tflite', 'wb') as f:
    f.write(tflite_int8)
print("INT8 quantized model saved as 'int8_quantized_model.tflite'")

# Lastly, Float16 Quantization

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_float16 = converter.convert()

with open('models/float16_quantized_model.tflite', 'wb') as f:
    f.write(tflite_float16)
print("Float16 quantized model saved as 'float16_quantized_model.tflite'")

# Confirmation

print("All TFLite variants created successfully!")

