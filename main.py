import os
import tensorflow as tf

train_dir = os.path.join('datasets', 'train')
test_dir = os.path.join('datasets', 'test')

train_dataset = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    label_mode='int',
    labels='inferred',
    follow_links=True
)

test_dataset = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    label_mode='int',
    labels='inferred',
    follow_links=True
)

# Vectorize training dataset
VOCAB_SIZE = 5000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

print(len(encoder.get_vocabulary()))

# Building Recurrent Neural Network
# Using GRU cells and Hyperbolic Tangent as activation function
cell = tf.keras.layers.GRUCell(30, activation='tanh')
model = tf.keras.models.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.RNN(cell)),
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-2),
                metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=10)

# Plotting the accuracy and loss
test_loss, test_acc = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

