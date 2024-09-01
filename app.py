
# import cv2
# import torch
# import numpy as np
# import imutils
# import pickle
# import os
# from torchvision import transforms
# import timm
# from flask import Flask, render_template, Response

# app = Flask(__name__)

# # Load model and other resources
# model_name = 'senet154'
# le_path = 'label_encoder.pickle'
# encodings = 'encoded_faces.pickle'
# detector_folder = 'face_detector'
# confidence = 0.5
# args = {
#     'le': le_path,
#     'detector': detector_folder,
#     'encodings': encodings,
#     'confidence': confidence
# }

# # Load the encoded faces and names
# with open(args['encodings'], 'rb') as file:
#     encoded_data = pickle.loads(file.read())

# # Load the face detector
# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# # Load the pretrained liveness detection model
# liveness_model = timm.create_model(model_name, pretrained=True, num_classes=2)
# liveness_model.eval()

# # Load the label encoder
# le = pickle.loads(open(args['le'], 'rb').read())

# # Define preprocessing transforms for the PyTorch model
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Initialize the webcam

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frm = imutils.resize(frame, width=800)
#             (h, w) = frm.shape[:2]
#             blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
#             detector_net.setInput(blob)
#             detections = detector_net.forward()

#             for i in range(0, detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]
                
#                 if confidence > args['confidence']:
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     (startX, startY, endX, endY) = box.astype('int')

#                     startX = max(0, startX - 20)
#                     startY = max(0, startY - 20)
#                     endX = min(w, endX + 20)
#                     endY = min(h, endY + 20)

#                     face = frm[startY:endY, startX:endX]

#                     try:
#                         face_tensor = preprocess(face)
#                     except Exception as e:
#                         print(f'[ERROR] {e}')
#                         continue

#                     face_tensor = face_tensor.unsqueeze(0)

#                     with torch.no_grad():
#                         preds = liveness_model(face_tensor).numpy()[0]

#                     j = np.argmax(preds)
#                     label_name = le.classes_[j]

#                     label = f'{label_name}: {preds[j]:.4f}'
#                     print(f'[INFO] {label_name}')

#                     if label_name == 'fake':
#                         cv2.putText(frm, "Fake Alert!", (startX, endY + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
#                         color = (0, 0, 255)
#                     else:
#                         color = (0, 255, 0)

#                     cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

#             ret, buffer = cv2.imencode('.jpg', frm)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start_video')
# def start_video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


# from streamlit_webrtc import webrtc_streamer
# import av
# import cv2
# import torch
# import numpy as np
# import imutils
# import pickle
# import os
# from torchvision import transforms
# import timm


# # Paths and parameters
# model_name = 'efficientnet_b0'  # Using SENet154 from timm
# le_path = 'label_encoder.pickle'
# encodings = 'encoded_faces.pickle'
# detector_folder = 'face_detector'
# confidence = 0.5
# args = {
#     'le': le_path,
#     'detector': detector_folder,
#     'encodings': encodings,
#     'confidence': confidence
# }

# # Load the encoded faces and names
# print('[INFO] loading encodings...')
# with open(args['encodings'], 'rb') as file:
#     encoded_data = pickle.loads(file.read())

# # Load our serialized face detector from disk
# print('[INFO] loading face detector...')
# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# # Load the pretrained liveness detection model using timm
# liveness_model = timm.create_model(model_name, pretrained=True, num_classes=2)  # Change num_classes to 2 for real/fake
# liveness_model.eval()  # Set the model to evaluation mode

# # Load the label encoder from disk
# le = pickle.loads(open(args['le'], 'rb').read())

# # Define preprocessing transforms for the PyTorch model
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Resize to match model input
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet
# ])

# class VideoProcessor:
#     def recv(self, frame):
#         frm = frame.to_ndarray(format="bgr24")

#         # Resize the frame
#         frm = imutils.resize(frm, width=800)
#         (h, w) = frm.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

#         # Pass the blob through the network and obtain the detections
#         detector_net.setInput(blob)
#         detections = detector_net.forward()

#         # Iterate over the detections
#         for i in range(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]

#             if confidence > args['confidence']:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype('int')

#                 startX = max(0, startX - 20)
#                 startY = max(0, startY - 20)
#                 endX = min(w, endX + 20)
#                 endY = min(h, endY + 20)

#                 face = frm[startY:endY, startX:endX]

#                 # Preprocess the face for the PyTorch model
#                 try:
#                     face_tensor = preprocess(face)
#                 except Exception as e:
#                     print(f'[ERROR] {e}')
#                     continue

#                 # Add a batch dimension
#                 face_tensor = face_tensor.unsqueeze(0)

#                 # Pass the face ROI through the trained liveness detection model
#                 with torch.no_grad():  # Disable gradient calculation
#                     preds = liveness_model(face_tensor).numpy()[0]

#                 j = np.argmax(preds)
#                 label_name = le.classes_[j]

#                 # Draw the label and bounding box on the frame
#                 label = f'{label_name}: {preds[j]:.4f}'
#                 print(f'[INFO] {label_name}')

#                 if label_name == 'fake':
#                     cv2.putText(frm, "Fake Alert!", (startX, endY + 25),
#                                 cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
#                     color = (0, 0, 255)  # Red for fake
#                 else:
#                     color = (0, 255, 0)  # Green for real

#                 cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                 cv2.putText(frm, label, (startX, startY - 10),
#                             cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                 cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

#         return av.VideoFrame.from_ndarray(frm, format='bgr24')

# webrtc_streamer(key="key", video_processor_factory=VideoProcessor, rtc_configuration={
#     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
# }, sendback_audio=False, video_receiver_size=1)  



# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import av
# import cv2
# import torch
# import numpy as np
# import imutils
# import pickle
# import os
# from torchvision import transforms
# import timm

# # Paths and parameters
# model_name = 'senet154'  # Using SENet154 from timm
# le_path = 'label_encoder.pickle'
# encodings = 'encoded_faces.pickle'
# detector_folder = 'face_detector'
# confidence = 0.5
# args = {
#     'le': le_path,
#     'detector': detector_folder,
#     'encodings': encodings,
#     'confidence': confidence
# }

# # Load the encoded faces and names
# print('[INFO] loading encodings...')
# with open(args['encodings'], 'rb') as file:
#     encoded_data = pickle.loads(file.read())

# # Load the face detector model
# print('[INFO] loading face detector...')
# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# # Load the pretrained liveness detection model using timm
# liveness_model = timm.create_model(model_name, pretrained=True, num_classes=2)
# liveness_model.eval()  # Set the model to evaluation mode

# # Load the label encoder
# le = pickle.loads(open(args['le'], 'rb').read())

# # Define preprocessing transforms
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Resize to match model input
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Streamlit UI enhancements
# st.title("Browser-Based Face Liveness Detection")
# st.write("""
#     This tool detects whether a face is 'Real' or 'Fake' to prevent spoofing attacks in a browser-based context.
# """)

# st.sidebar.header("Instructions")
# st.sidebar.write("""
# 1. Ensure your face is well-lit and visible to the camera.
# 2. Position yourself in front of the camera.
# 3. Click the **Start Detection** button to begin the liveness check.
# 4. Real-time feedback will be displayed below.
# """)

# st.sidebar.subheader("Controls")
# start_detection = st.sidebar.button("Start Detection")

# class VideoProcessor:
#     def recv(self, frame):
#         frm = frame.to_ndarray(format="bgr24")

#         # Resize the frame
#         frm = imutils.resize(frm, width=800)
#         (h, w) = frm.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

#         # Pass the blob through the network
#         detector_net.setInput(blob)
#         detections = detector_net.forward()

#         # Iterate over the detections
#         for i in range(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]

#             if confidence > args['confidence']:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype('int')

#                 startX = max(0, startX - 20)
#                 startY = max(0, startY - 20)
#                 endX = min(w, endX + 20)
#                 endY = min(h, endY + 20)

#                 face = frm[startY:endY, startX:endX]

#                 # Preprocess the face for the model
#                 try:
#                     face_tensor = preprocess(face)
#                 except Exception as e:
#                     print(f'[ERROR] {e}')
#                     continue

#                 face_tensor = face_tensor.unsqueeze(0)

#                 # Predict with the liveness detection model
#                 with torch.no_grad():
#                     preds = liveness_model(face_tensor).numpy()[0]

#                 j = np.argmax(preds)
#                 label_name = le.classes_[j]

#                 # Display results
#                 label = f'{label_name}: {preds[j]:.4f}'
#                 print(f'[INFO] {label_name}')

#                 if label_name == 'fake':
#                     cv2.putText(frm, "Fake Alert!", (startX, endY + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
#                     color = (0, 0, 255)  # Red for fake
#                 else:
#                     color = (0, 255, 0)  # Green for real

#                 cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                 cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                 cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

#         return av.VideoFrame.from_ndarray(frm, format='bgr24')

# if start_detection:
#     webrtc_streamer(key="key", video_processor_factory=VideoProcessor, rtc_configuration={
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     }, sendback_audio=False, video_receiver_size=1)
# else:
#     st.info("Click the **Start Detection** button to begin.")
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import av
# import cv2
# import torch
# import numpy as np
# import imutils
# import pickle
# import os
# from torchvision import transforms
# import timm

# # Paths and parameters
# model_name = 'senet154'
# le_path = 'label_encoder.pickle'
# encodings = 'encoded_faces.pickle'
# detector_folder = 'face_detector'
# confidence = 0.5
# args = {
#     'le': le_path,
#     'detector': detector_folder,
#     'encodings': encodings,
#     'confidence': confidence
# }

# # Load resources
# @st.cache_resource
# def load_resources():
#     print('[INFO] Loading resources...')
    
#     with open(args['encodings'], 'rb') as file:
#         encoded_data = pickle.load(file)

#     proto_path = os.path.join(args['detector'], 'deploy.prototxt')
#     model_path = os.path.join(args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel')
#     detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

#     liveness_model = timm.create_model(model_name, pretrained=True, num_classes=2)
#     liveness_model.eval()

#     le = pickle.load(open(args['le'], 'rb'))
    
#     preprocess = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     return detector_net, liveness_model, le, preprocess

# detector_net, liveness_model, le, preprocess = load_resources()

# # Streamlit UI
# st.title("Browser-Based Face Liveness Detection")
# st.write("""
#     This tool detects whether a face is 'Real' or 'Fake' to prevent spoofing attacks in a browser-based context.
# """)

# st.sidebar.header("Instructions")
# st.sidebar.write("""
# 1. Ensure your face is well-lit and visible to the camera.
# 2. Position yourself in front of the camera.
# 3. Click the **Start Detection** button to begin the liveness check.
# 4. Real-time feedback will be displayed below.
# """)

# st.sidebar.subheader("Controls")
# start_detection = st.sidebar.button("Start Detection")

# class VideoProcessor:
#     def recv(self, frame):
#         frm = frame.to_ndarray(format="bgr24")
#         frm = imutils.resize(frm, width=800)
#         (h, w) = frm.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
#         detector_net.setInput(blob)
#         detections = detector_net.forward()

#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > confidence:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype('int')
#                 startX = max(0, startX - 20)
#                 startY = max(0, startY - 20)
#                 endX = min(w, endX + 20)
#                 endY = min(h, endY + 20)

#                 face = frm[startY:endY, startX:endX]

#                 try:
#                     face_tensor = preprocess(face)
#                 except Exception as e:
#                     print(f'[ERROR] {e}')
#                     continue

#                 face_tensor = face_tensor.unsqueeze(0)
#                 with torch.no_grad():
#                     preds = liveness_model(face_tensor).numpy()[0]

#                 j = np.argmax(preds)
#                 label_name = le.classes_[j]

#                 label = f'{label_name}: {preds[j]:.4f}'
#                 print(f'[INFO] {label_name}')

#                 color = (0, 255, 0) if label_name == 'real' else (0, 0, 255)
#                 cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                 cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                 cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

#         return av.VideoFrame.from_ndarray(frm, format='bgr24')

# if start_detection:
#     webrtc_streamer(key="key", video_processor_factory=VideoProcessor, rtc_configuration={
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     }, sendback_audio=False, video_receiver_size=1)
# else:
#     st.info("Click the **Start Detection** button to begin.")



# 
# 
# import cv2
# import torch
# import numpy as np
# import imutils
# import pickle
# import os
# from torchvision import transforms
# from flask import Flask, render_template, Response
# from tensorflow.keras.models import load_model  # Import Keras load_model

# app = Flask(__name__)

# # Load model and other resources
# le_path = 'label_encoder.pickle'  # Path to your label encoder
# encodings = 'encoded_faces.pickle'  # Path to your encoded faces
# detector_folder = 'face_detector'  # Path to your face detector files
# confidence = 0.5  # Confidence threshold for face detection
# args = {
#     'le': le_path,
#     'detector': detector_folder,
#     'encodings': encodings,
#     'confidence': confidence
# }

# # Load the encoded faces and names
# with open(args['encodings'], 'rb') as file:
#     encoded_data = pickle.loads(file.read())

# # Load the face detector
# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# # Load your custom anti-spoofing model
# liveness_model = load_model('C:/Users/prata/OneDrive/Documents/sih/Face-Liveness-Detection-Anti-Spoofing-Web-App/finalyearproject_antispoofing_model_98-0.942647.h5')


# # Load the label encoder
# le = pickle.loads(open(args['le'], 'rb').read())

# # Define preprocessing transforms for the model
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Adjust this based on your model input size
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust if needed
# ])

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Initialize the webcam

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frm = imutils.resize(frame, width=800)
#             (h, w) = frm.shape[:2]
#             blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
#             detector_net.setInput(blob)
#             detections = detector_net.forward()

#             for i in range(0, detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]
                
#                 if confidence > args['confidence']:
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     (startX, startY, endX, endY) = box.astype('int')

#                     startX = max(0, startX - 20)
#                     startY = max(0, startY - 20)
#                     endX = min(w, endX + 20)
#                     endY = min(h, endY + 20)

#                     face = frm[startY:endY, startX:endX]

#                     try:
#                         face_tensor = preprocess(face).unsqueeze(0).numpy()  # Convert to NumPy array
#                     except Exception as e:
#                         print(f'[ERROR] {e}')
#                         continue

#                     # Predict liveness with your custom model
#                     preds = liveness_model.predict(face_tensor)
#                     label_name = 'real' if preds[0][0] > 0.5 else 'fake'

#                     label = f'{label_name}: {preds[0][0]:.4f}'
#                     print(f'[INFO] {label_name}')

#                     color = (0, 255, 0) if label_name == 'real' else (0, 0, 255)
#                     cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

#             ret, buffer = cv2.imencode('.jpg', frm)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start_video')
# def start_video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# import cv2
# import torch
# import numpy as np
# import imutils
# import pickle
# import os
# from torchvision import transforms
# import timm
# from flask import Flask, render_template, Response

# app = Flask(__name__)

# # Load model and other resources
# model_name = 'mobilenetv4_hybrid_large'
# le_path = 'label_encoder.pickle'
# encodings = 'encoded_faces.pickle'
# detector_folder = 'face_detector'
# confidence_threshold = 0.5  # Threshold for face detection confidence

# args = {
#     'le': le_path,
#     'detector': detector_folder,
#     'encodings': encodings,
#     'confidence': confidence_threshold
# }

# # Load the encoded faces and names
# with open(args['encodings'], 'rb') as file:
#     encoded_data = pickle.loads(file.read())

# # Load the face detector
# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# # Load the pretrained liveness detection model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
# liveness_model = timm.create_model(model_name, pretrained=True, num_classes=2)
# liveness_model.eval()
# liveness_model.to(device)  # Move model to the appropriate device

# # Load the label encoder
# le = pickle.loads(open(args['le'], 'rb').read())

# # Define preprocessing transforms for the PyTorch model
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Resize to match the model input size
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Initialize the webcam

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frm = imutils.resize(frame, width=800)  # Resize the frame for faster processing
#             (h, w) = frm.shape[:2]
#             blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
#             detector_net.setInput(blob)
#             detections = detector_net.forward()

#             for i in range(0, detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]
                
#                 if confidence > args['confidence']:
#                     # Calculate bounding box for detected face
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     (startX, startY, endX, endY) = box.astype('int')

#                     # Adjust bounding box for face extraction
#                     startX = max(0, startX - 20)
#                     startY = max(0, startY - 20)
#                     endX = min(w, endX + 20)
#                     endY = min(h, endY + 20)

#                     # Extract the face from the frame
#                     face = frm[startY:endY, startX:endX]

#                     try:
#                         face_tensor = preprocess(face).unsqueeze(0).to(device)  # Preprocess and move to device
#                     except Exception as e:
#                         print(f'[ERROR] {e}')
#                         continue

#                     with torch.no_grad():
#                         preds = liveness_model(face_tensor).cpu().numpy()[0]  # Get predictions from the model
                        
#                     # Determine the predicted label
#                     j = np.argmax(preds)
#                     label_name = le.classes_[j]
#                     label = f'{label_name}: {preds[j]:.4f}'
#                     print(f'[INFO] {label_name}')

#                     # Display results on the frame
#                     color = (0, 0, 255) if label_name == 'fake' else (0, 255, 0)  # Red for fake, Green for real
#                     cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode('.jpg', frm)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start_video')
# def start_video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)




# import torch
# import numpy as np
# import imutils
# import pickle
# import os
# from torchvision import transforms
# import timm
# from flask import Flask, render_template, Response
# import cv2

# app = Flask(__name__)

# # Load model and other resources
# model_name = 'DeepPixBis'
# le_path = 'label_encoder.pickle'
# encodings = 'encoded_faces.pickle'
# detector_folder = 'face_detector'
# confidence_threshold = 0.5  # Threshold for face detection confidence

# args = {
#     'le': le_path,
#     'detector': detector_folder,
#     'encodings': encodings,
#     'confidence': confidence_threshold
# }

# # Load the encoded faces and names
# with open(args['encodings'], 'rb') as file:
#     encoded_data = pickle.loads(file.read())

# # Load the face detector
# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# # Load the pretrained DeepPixBis model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
# liveness_model = torch.hub.load('pytorch/vision:v0.10.0', 'DeepPixBis', pretrained=True)
# liveness_model.eval()
# liveness_model.to(device)  # Move model to the appropriate device

# # Load the label encoder
# le = pickle.loads(open(args['le'], 'rb').read())

# # Define preprocessing transforms for the PyTorch model
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Resize to match the model input size
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Initialize the webcam

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frm = imutils.resize(frame, width=800)  # Resize the frame for faster processing
#             (h, w) = frm.shape[:2]
#             blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
#             detector_net.setInput(blob)
#             detections = detector_net.forward()

#             for i in range(0, detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]
                
#                 if confidence > args['confidence']:
#                     # Calculate bounding box for detected face
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     (startX, startY, endX, endY) = box.astype('int')

#                     # Adjust bounding box for face extraction
#                     startX = max(0, startX - 20)
#                     startY = max(0, startY - 20)
#                     endX = min(w, endX + 20)
#                     endY = min(h, endY + 20)

#                     # Extract the face from the frame
#                     face = frm[startY:endY, startX:endX]

#                     try:
#                         face_tensor = preprocess(face).unsqueeze(0).to(device)  # Preprocess and move to device
#                     except Exception as e:
#                         print(f'[ERROR] {e}')
#                         continue

#                     with torch.no_grad():
#                         preds = liveness_model(face_tensor).cpu().numpy()[0]  # Get predictions from the model
                        
#                     # Determine the predicted label
#                     j = np.argmax(preds)
#                     label_name = le.classes_[j]
#                     label = f'{label_name}: {preds[j]:.4f}'
#                     print(f'[INFO] {label_name}')

#                     # Display results on the frame
#                     color = (0, 0, 255) if label_name == 'fake' else (0, 255, 0)  # Red for fake, Green for real
#                     cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

#                     # Draw bounding box around the detected face
#                     cv2.rectangle(frm, (startX, startY), (endX, endY), color, 2)

#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode('.jpg', frm)
#             frame = buffer.tobytes()

#             # Yield the frame to the Flask response
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()  # Release the video capture object

# @app.route('/')
# def index():
#     return render_template('index.html')  # Render the index template

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)  # Run the Flask app

# import cv2
# import torch
# import numpy as np
# import imutils
# import pickle
# import os
# from torchvision import transforms
# import timm
# from flask import Flask, render_template, Response

# app = Flask(__name__)

# # Load model and other resources
# model_name = 'resnet50'
# le_path = 'label_encoder.pickle'
# encodings = 'encoded_faces.pickle'
# detector_folder = 'face_detector'
# confidence_threshold = 0.5 # Threshold for face detection confidence

# args = {
#     'le': le_path,
#     'detector': detector_folder,
#     'encodings': encodings,
#     'confidence': confidence_threshold
# }

# # Load the encoded faces and names
# with open(args['encodings'], 'rb') as file:
#     encoded_data = pickle.loads(file.read())

# # Load the face detector
# proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
# model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
# detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# # Load the pretrained liveness detection model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
# liveness_model = timm.create_model(model_name, pretrained=True, num_classes=2)
# liveness_model.eval()
# liveness_model.to(device)  # Move model to the appropriate device

# # Load the label encoder
# le = pickle.loads(open(args['le'], 'rb').read())

# # Define preprocessing transforms for the PyTorch model
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Resize to match the model input size
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Initialize the webcam

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             frm = imutils.resize(frame, width=800)  # Resize the frame for faster processing
#             (h, w) = frm.shape[:2]
#             blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
#             detector_net.setInput(blob)
#             detections = detector_net.forward()

#             for i in range(0, detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]
                
#                 if confidence > args['confidence']:
#                     # Calculate bounding box for detected face
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     (startX, startY, endX, endY) = box.astype('int')

#                     # Adjust bounding box for face extraction
#                     startX = max(0, startX - 20)
#                     startY = max(0, startY - 20)
#                     endX = min(w, endX + 20)
#                     endY = min(h, endY + 20)

#                     # Extract the face from the frame
#                     face = frm[startY:endY, startX:endX]

#                     try:
#                         face_tensor = preprocess(face).unsqueeze(0).to(device)  # Preprocess and move to device
#                     except Exception as e:
#                         print(f'[ERROR] {e}')
#                         continue

#                     with torch.no_grad():
#                         preds = liveness_model(face_tensor).cpu().numpy()[0]  # Get predictions from the model
                        
#                     # Determine the predicted label
#                     j = np.argmax(preds)
#                     label_name = le.classes_[j]
#                     label = f'{label_name}: {preds[j]:.4f}'
#                     print(f'[INFO] {label_name}')

#                     # Display results on the frame
#                     color = (0, 0, 255) if label_name == 'fake' else (0, 255, 0)  # Red for fake, Green for real
#                     cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
#                     cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode('.jpg', frm)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start_video')
# def start_video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

import cv2
import torch

import torch.nn as nn
import numpy as np
import imutils
import pickle
import os
from torchvision import transforms
import timm
from flask import Flask, render_template, Response,request,redirect,url_for,send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and other resources
# model_name = 'resnet50'
model_name = 'senet154'


le_path = 'label_encoder.pickle'
encodings = 'encoded_faces.pickle'
detector_folder = 'face_detector'
confidence_threshold = 0.5  # Threshold for face detection confidence


# # Load the entire model
# model_name = torch.load(r'C:\Users\prata\OneDrive\Documents\sih\Face-Liveness-Detection-Anti-Spoofing-Web-App\model2.pt')

# # Set the model to evaluation mode
# model_name.eval()


args = {
    'le': le_path,
    'detector': detector_folder,
    'encodings': encodings,
    'confidence': confidence_threshold
}

# Load the encoded faces and names
with open(args['encodings'], 'rb') as file:
    encoded_data = pickle.loads(file.read())

# Load the face detector
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Load the pretrained liveness detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
liveness_model = timm.create_model(model_name, pretrained=True, num_classes=2)
liveness_model.eval()
liveness_model.to(device)  # Move model to the appropriate device

# Load the label encoder
le = pickle.loads(open(args['le'], 'rb').read())

# Define preprocessing transforms for the PyTorch model
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match the model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_frames():
    cap = cv2.VideoCapture(0)  # Initialize the webcam

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frm = imutils.resize(frame, width=800)  # Resize the frame for faster processing
            (h, w) = frm.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            detector_net.setInput(blob)
            detections = detector_net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > args['confidence']:
                    # Calculate bounding box for detected face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')

                    # Adjust bounding box for face extraction
                    startX = max(0, startX - 20)
                    startY = max(0, startY - 20)
                    endX = min(w, endX + 20)
                    endY = min(h, endY + 20)

                    # Extract the face from the frame
                    face = frm[startY:endY, startX:endX]

                    try:
                        face_tensor = preprocess(face).unsqueeze(0).to(device)  # Preprocess and move to device
                    except Exception as e:
                        print(f'[ERROR] {e}')
                        continue

                    with torch.no_grad():
                        preds = liveness_model(face_tensor).cpu().numpy()[0]  # Get predictions from the model
                        
                    # Determine the predicted label
                    j = np.argmax(preds)
                    label_name = le.classes_[j]
                    label = f'{label_name}: {preds[j]:.4f}'
                    print(f'[INFO] {label_name}')

                    # Display results on the frame
                    color = (0, 0, 255) if label_name == 'fake' else (0, 255, 0)  # Red for fake, Green for real
                    cv2.putText(frm, label_name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
                    cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
                    cv2.rectangle(frm, (startX, startY), (endX, endY), color, 4)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frm)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    return redirect(url_for('upload_faces'))

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', title="About")
@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html', title="Contact")

@app.route('/upload_faces', methods=['POST', 'GET'])
def upload_faces():
    global content, graph
    # Display Page
    if request.method == 'GET':
        return render_template('upload_faces.html')
    
@app.route('/start_video')
def start_video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

