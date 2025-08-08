#!/usr/bin/env python3
"""
Script terpisah untuk verifikasi anti-spoof
Dijalankan sebagai subprocess untuk menghindari konflik dengan PyQt
"""

import cv2
import os
import random
import time
import numpy as np
import sys
from tensorflow.keras.models import load_model
import face_recognition

# Coba import gaze_tracking
try:
    from gaze_tracking import GazeTracking
    GAZE_TRACKING_AVAILABLE = True
except ImportError:
    GAZE_TRACKING_AVAILABLE = False

class SimpleVerificationSystem:
    def __init__(self):
        print("Memulai sistem verifikasi...")
        
        # Load model
        self.model = load_model("model_final.keras")
        print("✅ Model dimuat")
        
        # Load dataset
        self.label_names = self.load_label_names("split/test")
        print(f"✅ Dataset: {list(self.label_names.values())}")
        
        # Gaze tracking
        if GAZE_TRACKING_AVAILABLE:
            self.gaze = GazeTracking()
            self.use_gaze_tracking = True
            print("✅ GazeTracking aktif")
        else:
            self.use_gaze_tracking = False
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("✅ OpenCV fallback aktif")
        
        # Commands
        self.commands = ["LIHAT KIRI", "LIHAT KANAN", "LIHAT TENGAH", "KEDIP"]
        
        # State
        self.stage = "DETECTING_FACE"
        self.detected_name = None
        self.current_command = None
        self.command_time = None
        
        # Counters
        self.face_count = 0
        self.action_count = 0
        
        # OpenCV fallback variables
        self.eye_closed_frames = 0
        self.eye_open_frames = 0
        self.blink_detected = False
        
        # Timer
        self.start_time = time.time()
        self.timeout = 30
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Kamera tidak dapat diakses")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def load_label_names(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
        labels = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        if not labels:
            raise ValueError("Dataset kosong")
        return {i: label for i, label in enumerate(labels)}

    def detect_action_gaze(self, frame):
        self.gaze.refresh(frame)
        annotated = self.gaze.annotated_frame()
        if annotated is not None:
            frame[:] = annotated[:]
        
        if self.gaze.is_blinking():
            return "KEDIP"
        elif self.gaze.is_right():
            return "LIHAT KANAN"
        elif self.gaze.is_left():
            return "LIHAT KIRI"
        elif self.gaze.is_center():
            return "LIHAT TENGAH"
        return ""

    def detect_action_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            return ""
        
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(15, 15))
        
        # Draw crosses on eyes
        for (ex, ey, ew, eh) in eyes:
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            cv2.line(frame, (eye_center_x - 10, eye_center_y), (eye_center_x + 10, eye_center_y), (0, 255, 0), 2)
            cv2.line(frame, (eye_center_x, eye_center_y - 10), (eye_center_x, eye_center_y + 10), (0, 255, 0), 2)
        
        # Blink detection
        if len(eyes) == 0:
            self.eye_closed_frames += 1
            self.eye_open_frames = 0
            if self.eye_closed_frames >= 3 and not self.blink_detected:
                self.blink_detected = True
                return "KEDIP"
        elif len(eyes) >= 2:
            self.eye_open_frames += 1
            if self.eye_open_frames >= 2:
                self.eye_closed_frames = 0
                self.blink_detected = False
        
        # Head direction
        frame_center = frame.shape[1] // 2
        face_center = x + w // 2
        offset = face_center - frame_center
        
        if offset < -80:
            return "LIHAT KANAN"
        elif offset > 80:
            return "LIHAT KIRI"
        else:
            return "LIHAT TENGAH"

    def recognize_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return None, 0, None
        
        top, right, bottom, left = face_locations[0]
        face_image = frame[top:bottom, left:right]
        
        if face_image.size == 0:
            return None, 0, None
        
        resized_face = cv2.resize(face_image, (224, 224))
        normalized_face = resized_face.astype("float32") / 255.0
        face_input = np.expand_dims(normalized_face, axis=0)
        
        predictions = self.model.predict(face_input, verbose=0)
        confidence = np.max(predictions)
        label_idx = np.argmax(predictions)
        
        if confidence > 0.7:
            name = self.label_names.get(label_idx, "Unknown")
            return name, confidence, (left, top, right, bottom)
        
        return None, confidence, (left, top, right, bottom)

    def draw_ui(self, frame):
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        cv2.putText(frame, "VERIFIKASI ANTI-SPOOF", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Stage
        if self.stage == "DETECTING_FACE":
            text = "1. Mendeteksi wajah..."
        elif self.stage == "RECOGNIZING_FACE":
            text = "2. Mengenali identitas..."
        elif self.stage == "GIVING_COMMAND":
            text = f"3. Ikuti: {self.current_command}"
        elif self.stage == "VERIFYING_ACTION":
            text = f"4. Verifikasi: {self.current_command}"
        elif self.stage == "SUCCESS":
            text = "5. BERHASIL!"
        else:
            text = "Memulai..."
        
        cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Info
        if self.detected_name:
            cv2.putText(frame, f"Wajah: {self.detected_name}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Time
        remaining = max(0, self.timeout - (time.time() - self.start_time))
        cv2.putText(frame, f"Waktu: {int(remaining)}s", (frame.shape[1] - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Control
        cv2.putText(frame, "Tekan Q untuk batal", (20, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def process_stages(self, frame):
        # Face recognition
        name, confidence, location = self.recognize_face(frame)
        
        # Action detection
        if self.use_gaze_tracking:
            action = self.detect_action_gaze(frame)
        else:
            action = self.detect_action_opencv(frame)
        
        # Draw face box
        if location:
            left, top, right, bottom = location
            color = (0, 255, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            if name:
                cv2.putText(frame, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # State machine
        if self.stage == "DETECTING_FACE":
            if name and confidence > 0.7:
                self.face_count += 1
                self.detected_name = name
                if self.face_count >= 8:
                    print(f"✅ Wajah: {name}")
                    self.stage = "RECOGNIZING_FACE"
                    self.face_count = 0
            else:
                self.face_count = 0
                self.detected_name = None
        
        elif self.stage == "RECOGNIZING_FACE":
            if name == self.detected_name and confidence > 0.7:
                self.face_count += 1
                if self.face_count >= 5:
                    print(f"✅ Identitas: {self.detected_name}")
                    self.stage = "GIVING_COMMAND"
                    self.current_command = random.choice(self.commands)
                    self.command_time = time.time()
                    print(f"🎯 Perintah: {self.current_command}")
                    self.face_count = 0
            else:
                self.stage = "DETECTING_FACE"
                self.face_count = 0
        
        elif self.stage == "GIVING_COMMAND":
            if time.time() - self.command_time > 6:
                self.current_command = random.choice(self.commands)
                self.command_time = time.time()
                print(f"🔄 Perintah baru: {self.current_command}")
            
            if action == self.current_command:
                print(f"✅ Aksi: {action}")
                self.stage = "VERIFYING_ACTION"
                self.action_count = 0
        
        elif self.stage == "VERIFYING_ACTION":
            if action == self.current_command:
                self.action_count += 1
                if self.action_count >= 4:
                    print(f"🎉 BERHASIL!")
                    self.stage = "SUCCESS"
                    return True
            else:
                self.stage = "GIVING_COMMAND"
                self.action_count = 0
                self.command_time = time.time()
        
        # Show current action
        if action:
            cv2.putText(frame, f"Aksi: {action}", (frame.shape[1] - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return False

    def run(self):
        print("🚀 MULAI VERIFIKASI")
        print("Tahap: Deteksi → Pengenalan → Perintah → Verifikasi → Sukses")
        print("Tekan Q untuk batal\n")
        
        success = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if time.time() - self.start_time > self.timeout:
                    print("⏰ Timeout!")
                    break
                
                success = self.process_stages(frame)
                self.draw_ui(frame)
                
                cv2.imshow('Verifikasi Anti-Spoof', frame)
                
                if success:
                    time.sleep(2)
                    break
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("❌ Dibatalkan")
                    break
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
        
        return success

def main():
    try:
        system = SimpleVerificationSystem()
        result = system.run()
        
        # Exit code: 0 = success, 1 = failed
        sys.exit(0 if result else 1)
        
    except Exception as e:
        print(f"💥 Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()