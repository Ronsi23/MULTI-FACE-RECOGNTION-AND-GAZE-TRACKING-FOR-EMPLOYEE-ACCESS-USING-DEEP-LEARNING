from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QFrame, QGridLayout, QLineEdit,
                            QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import os
import shutil
import random
from datetime import datetime

class RegistrasiWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jendela Registrasi")
        self.setFixedSize(800, 600)
        
        # Initialize variables
        self.camera_active = False
        self.capture_timer = None
        self.cap = None
        self.current_name = ""
        self.captured_images_count = 0
        self.total_images_to_capture = 500
        self.dataset_path = "split"
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create left side (Image frame)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Create image frame
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.Box | QFrame.Plain)
        image_frame.setMinimumSize(400, 400)
        image_layout = QVBoxLayout(image_frame)
        
        # Create image label to display camera feed
        self.image_label = QLabel("Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(380, 380)
        image_layout.addWidget(self.image_label)
        
        left_layout.addWidget(image_frame)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        self.rekam_btn = QPushButton("Rekam")
        self.menu_btn = QPushButton("Menu")
        
        # Make Rekam button stretch
        self.rekam_btn.setSizePolicy(self.rekam_btn.sizePolicy().horizontalPolicy(),
                                  self.rekam_btn.sizePolicy().verticalPolicy())
        
        # Connect buttons
        self.rekam_btn.clicked.connect(self.toggle_camera)
        self.menu_btn.clicked.connect(self.close_camera_and_window)
        
        buttons_layout.addWidget(self.rekam_btn, stretch=2)
        buttons_layout.addWidget(self.menu_btn)
        
        left_layout.addLayout(buttons_layout)
        
        # Create right side (Details)
        right_widget = QWidget()
        right_layout = QGridLayout(right_widget)
        
        # Add date and time with current values
        current_datetime = datetime.now()
        
        right_layout.addWidget(QLabel("Date:"), 0, 0)
        self.date_label = QLabel(current_datetime.strftime("%Y-%m-%d"))
        right_layout.addWidget(self.date_label, 0, 1)
        
        right_layout.addWidget(QLabel("Time:"), 1, 0)
        self.time_label = QLabel(current_datetime.strftime("%H:%M:%S"))
        right_layout.addWidget(self.time_label, 1, 1)
        
        # Add name input field
        right_layout.addWidget(QLabel("Detail"), 3, 0)
        right_layout.addWidget(QLabel("Nama:"), 4, 0)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Masukkan nama")
        right_layout.addWidget(self.name_input, 4, 1)
        
        # Add status fields
        right_layout.addWidget(QLabel("Total Wajah:"), 5, 0)
        self.total_faces_label = QLabel("0")
        right_layout.addWidget(self.total_faces_label, 5, 1)
        
        right_layout.addWidget(QLabel("Status:"), 6, 0)
        self.status_label = QLabel("-")
        right_layout.addWidget(self.status_label, 6, 1)
        
        # Add widgets to main layout
        main_layout.addWidget(left_widget, stretch=2)
        main_layout.addWidget(right_widget, stretch=1)
        
        # Setup timer for updating time
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # Update every second
    
    def update_time(self):
        """Update the time label with current time"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)
    
    def toggle_camera(self):
        """Toggle camera on/off and start/stop face registration"""
        if not self.camera_active:
            # Check if name is provided
            self.current_name = self.name_input.text().strip()
            if not self.current_name:
                QMessageBox.warning(self, "Input Error", "Silakan masukkan nama terlebih dahulu!")
                return
            
            # Start camera
            self.start_camera()
        else:
            # Stop camera
            self.stop_camera()
    
    def start_camera(self):
        """Start camera and face registration process"""
        # Initialize capture and variables
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Tidak dapat mengakses kamera!")
            return
        
        # Create dataset directory for user
        self.user_path = os.path.join(self.dataset_path, self.current_name)
        os.makedirs(self.user_path, exist_ok=True)
        
        # Update UI
        self.camera_active = True
        self.captured_images_count = 0
        self.rekam_btn.setText("Berhenti")
        self.status_label.setText(f"Merekam di {self.user_path}")
        
        # Start capture timer
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.process_frame)
        self.capture_timer.start(30)  # 30ms -> ~33 fps
    
    def process_frame(self):
        """Process each frame from camera and detect faces"""
        if not self.cap or not self.camera_active:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        display_frame = frame.copy()

        if len(faces) > 0 and self.captured_images_count < self.total_images_to_capture:
            for (x, y, w, h) in faces:
                # Gambar rectangle
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Ambil wajah berwarna
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (224, 224))

                img_path = os.path.join(
                    self.user_path,
                    f"{self.current_name}_{self.captured_images_count}.jpg"
                )
                cv2.imwrite(img_path, face_img)

                self.captured_images_count += 1
                self.total_faces_label.setText(str(self.captured_images_count))

                if self.captured_images_count >= self.total_images_to_capture:
                    self.finalize_registration()
                    break

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio
        ))

    
    def finalize_registration(self):
        """Split dataset and clean up after registration is complete"""
        # Stop camera
        self.stop_camera(auto_finalize=True)
        
        # Split dataset into train/valid/test
        self.split_dataset()
        
        # Update status
        self.status_label.setText("Registrasi selesai!")
    
    def split_dataset(self):
        """Split the dataset into train, valid, and test sets"""
        try:
            # Create destination directories
            train_path = os.path.join(self.dataset_path, "train", self.current_name)
            valid_path = os.path.join(self.dataset_path, "valid", self.current_name)
            test_path = os.path.join(self.dataset_path, "test", self.current_name)
            
            for path in [train_path, valid_path, test_path]:
                os.makedirs(path, exist_ok=True)
            
            # Get all images
            images = [f for f in os.listdir(self.user_path) if f.endswith(".jpg")]
            
            if not images:
                self.status_label.setText("Tidak ada gambar yang tersimpan!")
                return
            
            # Shuffle images
            random.shuffle(images)
            
            # Split images
            total_images = len(images)
            train_split = int(0.7 * total_images)
            valid_split = int(0.2 * total_images)
            
            # Move images to respective folders
            for i, img in enumerate(images):
                src_path = os.path.join(self.user_path, img)
                
                if i < train_split:
                    dest_path = os.path.join(train_path, img)
                elif i < train_split + valid_split:
                    dest_path = os.path.join(valid_path, img)
                else:
                    dest_path = os.path.join(test_path, img)
                
                shutil.copy2(src_path, dest_path)  # Use copy2 instead of move to avoid issues
            
            # Remove original folder after successful copies
            shutil.rmtree(self.user_path)
            
            # Update status
            status_text = f"Dataset dibagi: Train({train_split}), Valid({valid_split}), Test({total_images-train_split-valid_split})"
            self.status_label.setText(status_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error splitting dataset: {str(e)}")
    
    def stop_camera(self, auto_finalize=False):
        """Stop the camera and clean up resources"""
        if self.capture_timer:
            self.capture_timer.stop()
        
        if self.cap:
            self.cap.release()
        
        self.camera_active = False
        self.rekam_btn.setText("Rekam")
        
        # Reset image
        self.image_label.setText("Image")
        self.image_label.setPixmap(QPixmap())
        
        # If stopped manually and we have images, ask to finalize
        if not auto_finalize and self.captured_images_count > 0:
            reply = QMessageBox.question(
                self, 'Konfirmasi', 
                f'Ada {self.captured_images_count} wajah terekam. Proses dataset?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.split_dataset()
    
    def close_camera_and_window(self):
        """Properly close camera before closing window"""
        self.stop_camera()
        self.close()
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_camera()
        event.accept()