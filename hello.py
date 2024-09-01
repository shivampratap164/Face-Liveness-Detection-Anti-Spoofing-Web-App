# import tensorflow as tf
# from keras.models import load_model
# import tf2onnx
# from keras.models import save_model

# # Assuming 'model' is your trained model
# from keras.layers import TFSMLayer
# import tensorflow as tf
# model = TFSMLayer('C:\\Users\\prata\\OneDrive\\Documents\\sih\\Face-Liveness-Detection-Anti-Spoofing-Web-App\\liveness.model', call_endpoint='serving_default')
# # Assuming 'model' is your trained Keras model
# tf.saved_model.save(model, 'C:\\Users\\prata\\OneDrive\\Documents\\sih\\Face-Liveness-Detection-Anti-Spoofing-Web-App\\liveness.model')
# # Load the model using TFSMLayer
# model = TFSMLayer('C:\\Users\\prata\\OneDrive\\Documents\\sih\\Face-Liveness-Detection-Anti-Spoofing-Web-App\\liveness.model', call_endpoint='serving_default')
# save_model(model, 'C:\\Users\\prata\\OneDrive\\Documents\\sih\\Face-Liveness-Detection-Anti-Spoofing-Web-App\\liveness.h5')
# # Load a pre-trained Keras model
# model = load_model('C:\\Users\\prata\\OneDrive\\Documents\\sih\\Face-Liveness-Detection-Anti-Spoofing-Web-App\\liveness.model')

# # Convert the Keras model to ONNX format
# onnx_file_path = "C:\\Face-Liveness-Detection-Anti-Spoofing-Web-App\\liveness_detection_model.onnx"
# spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)  # Adjust input shape as needed
# model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

# # Save the ONNX model to a file
# with open(onnx_file_path, "wb") as f:
#     f.write(model_proto.SerializeToString())

# print(f'Model has been converted to ONNX and saved at {onnx_file_path}')

# from keras.models import load_model

# # Load the model
# model = load_model('C:\\Users\\prata\\OneDrive\\Documents\\sih\\Face-Liveness-Detection-Anti-Spoofing-Web-App\\liveness.model')

# import tensorflow as tf
# import tf2onnx

# # Define the ONNX file path
# onnx_file_path = "C:\\Face-Liveness-Detection-Anti-Spoofing-Web-App\\liveness_detection_model.onnx"

# # Define input signature (adjust shape as needed)
# spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)  # Adjust input shape as needed

# # Convert the Keras model to ONNX format
# model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

# # Save the ONNX model to a file
# with open(onnx_file_path, "wb") as f:
#     f.write(model_proto.SerializeToString())

# print(f'Model has been converted to ONNX and saved at {onnx_file_path}')
import tensorflow as tf
import tf2onnx
from keras.models import load_model

# Load the Keras model using the correct file path
model = load_model(r'C:\Users\prata\OneDrive\Documents\sih\Face-Liveness-Detection-Anti-Spoofing-Web-App\liveness.h5')

# Define the ONNX file path
onnx_file_path = r"C:\Face-Liveness-Detection-Anti-Spoofing-Web-App\liveness_detection_model.onnx"

# Define the input signature (adjust shape as needed)
input_signature = [tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="input")]

# Convert the Keras model to ONNX format
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

# Save the ONNX model to a file
with open(onnx_file_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f'Model has been converted to ONNX and saved at {onnx_file_path}')