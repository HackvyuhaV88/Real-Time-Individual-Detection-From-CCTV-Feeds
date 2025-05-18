#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import pickle
import dlib
import numpy as np
import face_recognition
import threading
import datetime
import logging
from flask import Flask, jsonify, render_template
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.knn_clf = None
        self.trackers = []
        self.cameras = []
        self.running = False
        self.log_file = "face_recognition.log"
        self.known_faces_dir = "known_faces"
        self.history_size = 100
        
        # Initialize system components
        self.setup_logging()
        self.load_known_faces()
        
        # Add your custom face image from the path here
        self.add_face_from_path(r"C:\Users\amayg\Downloads\dip.jpeg", "Dipak")
        self.add_face_from_path(r"C:\Users\amayg\Downloads\WhatsApp Image 2025-05-17 at 4.12.18 PM.jpeg", "Amay")


        self.load_knn_model()
        
        logging.info("System initialized successfully")

    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_known_faces(self):
        try:
            if not os.path.exists(self.known_faces_dir):
                os.makedirs(self.known_faces_dir)
                return

            for filename in os.listdir(self.known_faces_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(self.known_faces_dir, filename)
                    image = face_recognition.load_image_file(path)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        self.known_face_encodings.append(encoding[0])
                        self.known_face_names.append(os.path.splitext(filename)[0])
            
            logging.info(f"Loaded {len(self.known_face_names)} known faces")
        except Exception as e:
            logging.error(f"Error loading known faces: {str(e)}")

    def add_face_from_path(self, image_path, name):
        """
        Loads an image from the specified path, encodes the face,
        and adds it to known faces list with the given name.
        """
        try:
            if os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(name)
                    logging.info(f"Added face encoding for {name} from {image_path}")
                else:
                    logging.warning(f"No face found in the image {image_path}")
            else:
                logging.error(f"Image path does not exist: {image_path}")
        except Exception as e:
            logging.error(f"Error adding face from path {image_path}: {str(e)}")

    def load_knn_model(self, model_path="trained_knn_model.clf"):
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.knn_clf = pickle.load(f)
                logging.info("Loaded KNN classifier model")
        except Exception as e:
            logging.error(f"Error loading KNN model: {str(e)}")

    def start_cameras(self, camera_ids=[0, 1]):
        try:
            self.running = True
            for cam_id in camera_ids:
                thread = threading.Thread(
                    target=self.process_camera_feed,
                    args=(cam_id,)
                )
                thread.daemon = True
                thread.start()
                self.cameras.append(thread)
            logging.info(f"Started {len(camera_ids)} camera feeds")
        except Exception as e:
            logging.error(f"Error starting cameras: {str(e)}")

    def process_camera_feed(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logging.error(f"Could not open camera {camera_id}")
            return

        process_frame = True
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Alternate between face detection and tracking
            if process_frame:
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                
                # Update trackers with new detections
                self.update_trackers(frame, face_locations)
                
                # Recognize faces
                self.recognize_faces(frame, face_encodings)

            # Show tracking information
            self.draw_trackers(frame)
            
            # Display frame
            cv2.imshow(f'Camera {camera_id}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            process_frame = not process_frame  # Process every other frame

        cap.release()
        cv2.destroyAllWindows()

    def update_trackers(self, frame, face_locations):
        try:
            self.trackers = []
            for (top, right, bottom, left) in face_locations:
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(left, top, right, bottom)
                tracker.start_track(frame, rect)
                self.trackers.append(tracker)
        except Exception as e:
            logging.error(f"Tracker update error: {str(e)}")

    def draw_trackers(self, frame):
        try:
            for tracker in self.trackers:
                position = tracker.get_position()
                left = int(position.left())
                top = int(position.top())
                right = int(position.right())
                bottom = int(position.bottom())
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        except Exception as e:
            logging.error(f"Tracker drawing error: {str(e)}")

    def recognize_faces(self, frame, face_encodings):
        try:
            for face_encoding in face_encodings:
                if self.knn_clf:
                    name = self.knn_recognition(face_encoding)
                else:
                    name = self.direct_comparison(face_encoding)
                
                self.log_recognition(name)
        except Exception as e:
            logging.error(f"Recognition error: {str(e)}")

    def knn_recognition(self, face_encoding):
        try:
            closest_distances = self.knn_clf.kneighbors([face_encoding], n_neighbors=1)
            if closest_distances[0][0][0] <= 0.4:
                return self.knn_clf.predict([face_encoding])[0]
            return "Unknown"
        except Exception as e:
            logging.error(f"KNN recognition error: {str(e)}")
            return "Unknown"

    def direct_comparison(self, face_encoding):
        try:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=0.5
            )
            if True in matches:
                first_match = matches.index(True)
                distance = np.linalg.norm(self.known_face_encodings[first_match] - face_encoding)
                if distance < 0.5:
                    return self.known_face_names[first_match]
            return "Unknown"
        except Exception as e:
            logging.error(f"Direct comparison error: {str(e)}")
            return "Unknown"

    def log_recognition(self, name):
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - Recognized: {name}"
            logging.info(log_entry)
            
            # Save to history file
            with open("recognition_history.txt", "a") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            logging.error(f"Logging error: {str(e)}")

    def stop_system(self):
        self.running = False
        for thread in self.cameras:
            thread.join()
        logging.info("System stopped")


# Flask Routes
fr_system = FaceRecognitionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_system():
    fr_system.start_cameras()
    return jsonify({"status": "System started"})

@app.route('/stop', methods=['POST'])
def stop_system():
    fr_system.stop_system()
    return jsonify({"status": "System stopped"})

@app.route('/logs')
def get_logs():
    try:
        if os.path.exists("recognition_history.txt"):
            with open("recognition_history.txt", "r") as f:
                logs = f.read()
        else:
            logs = "No logs available."
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"logs": f"Error reading logs: {str(e)}"})


if __name__ == '__main__':
    os.makedirs("known_faces", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
