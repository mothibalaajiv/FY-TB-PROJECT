import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape):

    model = models.Sequential([

        layers.Input(shape=input_shape),

        layers.Conv2D(
            16,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(
            64,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.Flatten(),

        layers.Dense(
            64,
            activation='relu'
        ),

        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss='binary_crossentropy',

        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
        ]
    )

    return model


def get_callbacks():

    return [

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
