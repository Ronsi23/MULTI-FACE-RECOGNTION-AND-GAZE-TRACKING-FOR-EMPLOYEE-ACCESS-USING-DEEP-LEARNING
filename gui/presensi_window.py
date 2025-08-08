from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFrame, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from datetime import datetime
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PresensiWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jendela Presensi")
        self.setFixedSize(800, 600)
        self.min_detection_frames = 5  # Lebih tinggi = lebih stabil tapi lambat
        
        # Inisialisasi variabel untuk face recognition
        self.face_recognition_active = False
        self.cap = None
        self.model = None
        self.index_to_class = {}
        self.face_cascade = None
        self.presensi_type = ""  # Untuk menyimpan jenis presensi (masuk/keluar)
        self.presensi_ready = False  # Flag untuk menandai apakah sedang dalam mode presensi
        
        # Load model dan setup face recognition
        self.setup_face_recognition()
        
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
        
        # Add "Image" label in the center of frame
        self.image_label = QLabel("Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        
        left_layout.addWidget(image_frame)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        self.presensi_keluar_btn = QPushButton("Presensi Keluar")
        self.presensi_masuk_btn = QPushButton("Presensi Masuk")
        self.menu_btn = QPushButton("Menu")
        
        buttons_layout.addWidget(self.presensi_keluar_btn)
        buttons_layout.addWidget(self.presensi_masuk_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.menu_btn)
        
        left_layout.addLayout(buttons_layout)
        
        # Create right side (Details)
        right_widget = QWidget()
        right_layout = QGridLayout(right_widget)
        
        # Add date and time
        right_layout.addWidget(QLabel("Date:"), 0, 0)
        self.date_label = QLabel(datetime.now().strftime("%Y-%m-%d"))
        right_layout.addWidget(self.date_label, 0, 1)
        
        right_layout.addWidget(QLabel("Time:"), 1, 0)
        self.time_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        right_layout.addWidget(self.time_label, 1, 1)
        
        # Add detail section
        right_layout.addWidget(QLabel("Detail"), 3, 0)
        right_layout.addWidget(QLabel("Nama:"), 4, 0)
        self.nama_label = QLabel("-")
        right_layout.addWidget(self.nama_label, 4, 1)
        
        right_layout.addWidget(QLabel("Arah Pandang:"), 5, 0)
        self.pandang_label = QLabel("-")
        right_layout.addWidget(self.pandang_label, 5, 1)
        
        right_layout.addWidget(QLabel("Status:"), 6, 0)
        self.status_label = QLabel("Kamera Aktif - Pilih jenis presensi")
        right_layout.addWidget(self.status_label, 6, 1)
        
        # Add widgets to main layout
        main_layout.addWidget(left_widget, stretch=2)
        main_layout.addWidget(right_widget, stretch=1)
        
        # Connect buttons
        self.menu_btn.clicked.connect(self.close_camera_and_window)
        self.presensi_masuk_btn.clicked.connect(self.set_presensi_masuk)
        self.presensi_keluar_btn.clicked.connect(self.set_presensi_keluar)
        
        # Setup timer for updating time
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # Update every second
        
        # Setup timer for face recognition
        self.face_recognition_timer = QTimer(self)
        self.face_recognition_timer.timeout.connect(self.process_frame)
        
        # PERUBAHAN UTAMA: Langsung start kamera saat window dibuka
        self.start_camera_preview()
        
    def setup_face_recognition(self):
        """Load model and setup face recognition"""
        try:
            # Load model
            self.model = load_model("model_final.keras")
            
            # Load class indices dari folder training
            datagen = ImageDataGenerator(rescale=1./255)
            temp_gen = datagen.flow_from_directory(
                "split/train",
                target_size=(224, 224),
                batch_size=1,
                class_mode='categorical',
                shuffle=False
            )
            self.index_to_class = {v: k for k, v in temp_gen.class_indices.items()}
            self.index_to_class[-1] = "Unknown"  # Tambahkan kelas unknown
            
            # Load Haar Cascade untuk deteksi wajah
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal memuat model face recognition: {str(e)}")
    
    def start_camera_preview(self):
        """Start camera preview immediately when window opens"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Tidak dapat mengakses kamera!")
            return
        
        self.face_recognition_active = True
        self.face_recognition_timer.start(30)  # ~30ms per frame
        self.status_label.setText("Kamera Aktif - Pilih jenis presensi")
    
    def set_presensi_masuk(self):
        """Set mode presensi masuk"""
        self.presensi_type = "MASUK"
        self.presensi_ready = True
        self.status_label.setText(f"Mode Presensi {self.presensi_type} - Arahkan wajah ke kamera")
        
        # Ubah warna tombol untuk menandai yang aktif
        self.presensi_masuk_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.presensi_keluar_btn.setStyleSheet("")
    
    def set_presensi_keluar(self):
        """Set mode presensi keluar"""
        self.presensi_type = "KELUAR"
        self.presensi_ready = True
        self.status_label.setText(f"Mode Presensi {self.presensi_type} - Arahkan wajah ke kamera")
        
        # Ubah warna tombol untuk menandai yang aktif
        self.presensi_keluar_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.presensi_masuk_btn.setStyleSheet("")
    
    def process_frame(self):
        """Process frame for face recognition"""
        if not self.face_recognition_active or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_face_recognition()
            return
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        display_frame = frame.copy()
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Selalu tampilkan bounding box untuk semua wajah yang terdeteksi
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Hanya lakukan prediksi jika mode presensi sudah dipilih
                if self.presensi_ready:
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (224, 224))
                    face_array = img_to_array(face_img) / 255.0
                    face_array = np.expand_dims(face_array, axis=0)
                    
                    preds = self.model.predict(face_array)
                    pred_index = np.argmax(preds)
                    confidence = np.max(preds)
                    
                    if confidence < 0.70:
                        label = "Unknown"
                        self.nama_label.setText("-")
                        self.pandang_label.setText("-")
                        self.status_label.setText(f"Presensi {self.presensi_type}: Wajah tidak dikenali")
                        cv2.putText(display_frame, "Unknown", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        label = f"{self.index_to_class[pred_index]}: {confidence*100:.2f}%"
                        self.nama_label.setText(self.index_to_class[pred_index])
                        self.pandang_label.setText("Hadap Depan")
                        self.status_label.setText(f"Presensi {self.presensi_type}: {self.index_to_class[pred_index]} - Tekan Enter untuk konfirmasi")
                        
                        cv2.putText(display_frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Auto-save presensi setelah 2 detik deteksi stabil (opsional)
                        # Atau bisa ditambahkan tombol konfirmasi
                        
                else:
                    # Tampilkan "Wajah Terdeteksi" jika belum pilih mode presensi
                    cv2.putText(display_frame, "Wajah Terdeteksi", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            # Reset info jika tidak ada wajah
            if not self.presensi_ready:
                self.nama_label.setText("-")
                self.pandang_label.setText("-")
        
        # Convert frame to QImage and display
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio
        ))
    
    def save_presensi(self, nama, jenis_presensi):
        """Simpan data presensi ke file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            presensi_data = f"{timestamp},{nama},{jenis_presensi}\n"
            
            # Buat folder presensi jika belum ada
            os.makedirs("presensi", exist_ok=True)
            
            # Simpan ke file CSV
            with open("presensi/data_presensi.csv", "a") as f:
                f.write(presensi_data)
                
            QMessageBox.information(self, "Sukses", 
                                  f"Presensi {jenis_presensi} berhasil dicatat untuk {nama}")
            
            # Reset mode presensi setelah berhasil
            self.reset_presensi_mode()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Gagal menyimpan presensi: {str(e)}")
    
    def reset_presensi_mode(self):
        """Reset mode presensi ke kondisi awal"""
        self.presensi_ready = False
        self.presensi_type = ""
        self.status_label.setText("Kamera Aktif - Pilih jenis presensi")
        
        # Reset warna tombol
        self.presensi_masuk_btn.setStyleSheet("")
        self.presensi_keluar_btn.setStyleSheet("")
        
        # Reset info
        self.nama_label.setText("-")
        self.pandang_label.setText("-")
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Konfirmasi presensi jika ada nama yang terdeteksi
            if self.presensi_ready and self.nama_label.text() != "-":
                self.save_presensi(self.nama_label.text(), self.presensi_type)
        super().keyPressEvent(event)
    
    def stop_face_recognition(self):
        """Stop face recognition and release camera"""
        if self.face_recognition_timer.isActive():
            self.face_recognition_timer.stop()
            
        if self.cap:
            self.cap.release()
            
        self.face_recognition_active = False
        self.image_label.clear()
        self.image_label.setText("Image")
    
    def update_time(self):
        """Update the time label with current time"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)
        self.date_label.setText(datetime.now().strftime("%Y-%m-%d"))
    
    def close_camera_and_window(self):
        """Properly close camera before closing window"""
        self.stop_face_recognition()
        self.close()
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_face_recognition()
        event.accept()