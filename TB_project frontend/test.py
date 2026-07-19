import tensorflow as tf

print("TensorFlow:", tf.__version__)

try:
    model = tf.keras.models.load_model("tb_cough_base_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error:")
    print(type(e).__name__)
    print(e)