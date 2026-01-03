import tensorflow as tf


def build_cnn(input_shape):
    """
    Build a 1D CNN for beat-level PVC detection.

    Parameters
    ----------
    input_shape : tuple
        Shape of input beat (samples, channels)

    Returns
    -------
    model : tf.keras.Model
        Compiled CNN model
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=5,
            activation='relu'
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            activation='relu'
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
