from flask import Flask, jsonify, render_template, send_from_directory
import os
import cv2
import threading
import datetime
import logging
import face_recognition
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = {}

        self.cameras = []
        self.running = False
        self.log_file = "face_recognition.log"
        self.known_faces_dir = r"C:\Users\amayg\Downloads\hackvyuha\known_faces"

        self.model_path = "yolov8n.pt"

        self.setup_logging()
        self.load_known_faces()

        # Load extra known faces manually (optional)



        self.yolo_model = YOLO(self.model_path)
        print("[INFO] YOLOv8 model loaded.")

    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_known_faces(self):
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            return

        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        path = os.path.join(person_dir, filename)
                        image = face_recognition.load_image_file(path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            if person_name not in self.known_faces:
                                self.known_faces[person_name] = []
                            self.known_faces[person_name].append(encodings[0])
                            print(f"[INFO] Loaded face for {person_name} from {filename}")



    def add_known_face(self, path, name):
        if os.path.exists(path):
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                self.known_face_encodings.append(encoding[0])
                self.known_face_names.append(name)
                print(f"[INFO] Added known face: {name}")
            else:
                print(f"[WARNING] No face found in {path}")
        else:
            print(f"[ERROR] File not found: {path}")

    def start_cameras(self, camera_ids=[0]):
        if self.running:
            print("[WARNING] System already running.")
            return
        self.running = True
        self.cameras = []
        for cam_id in camera_ids:
            thread = threading.Thread(target=self.process_camera_feed, args=(cam_id,))
            thread.daemon = True
            thread.start()
            self.cameras.append(thread)
        print("[INFO] Camera threads started.")

    def stop_system(self):
        self.running = False
        for thread in self.cameras:
            thread.join()
        print("[INFO] System stopped.")

    def stop_system(self):
        if not self.running:
            print("[WARNING] System not running.")
            return
        self.running = False
        print("[INFO] Stopping system. Waiting for camera threads to finish...")
        for thread in self.cameras:
            if thread.is_alive():
                thread.join()
        print("[INFO] System stopped.")

    def process_camera_feed(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {camera_id}")
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    face_locations = self.detect_faces_yolo(frame)
                except Exception as e:
                    print(f"[ERROR] YOLOv8 detection failed: {e}")
                    face_locations = face_recognition.face_locations(rgb_frame)

                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                self.recognize_faces(frame, face_encodings, face_locations)

                cv2.imshow(f"Camera {camera_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        except Exception as e:
            print(f"[ERROR] Exception in camera feed: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Camera {camera_id} released and windows closed.")


    def detect_faces_yolo(self, frame):
        results = self.yolo_model.predict(source=frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        face_locations = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_locations.append((y1, x2, y2, x1))  # top, right, bottom, left
        return face_locations

    def recognize_faces(self, frame, face_encodings, face_locations):
        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name = "Unknown"
            for person_name, encodings in self.known_faces.items():
                matches = face_recognition.compare_faces(encodings, encoding, tolerance=0.6)
                if True in matches:
                    name = person_name
                    break

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            self.log_recognition(name)


    def log_recognition(self, name):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} - Recognized: {name}"
        logging.info(entry)
        with open("recognition_history.txt", "a") as f:
            f.write(entry + "\n")


# ---------------- Flask Routes ----------------
fr_system = FaceRecognitionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    fr_system.start_cameras()
    return jsonify({"status": "System started"})

@app.route('/stop', methods=['POST'])
def stop():
    fr_system.stop_system()
    return jsonify({"status": "System stopped"})

@app.route('/logs')
def logs():
    if os.path.exists("recognition_history.txt"):
        with open("recognition_history.txt", "r") as f:
            content = f.read()
    else:
        content = "No logs available."
    return jsonify({"logs": content})

@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    os.makedirs("known_faces", exist_ok=True)
    app.run(debug=True)