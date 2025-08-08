import os
import sys
import subprocess
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QMessageBox) 
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# Import window lain di level modul (bukan di dalam method)
try:
    from .presensi_window import PresensiWindow
    from .registrasi_window import RegistrasiWindow
    from .training_window import TrainingWindow
except ImportError:
    from presensi_window import PresensiWindow
    from registrasi_window import RegistrasiWindow
    from training_window import TrainingWindow

# Import sistem verifikasi
try:
    from spoof import run_complete_verification
    VERIFICATION_AVAILABLE = True
except ImportError:
    print("Warning: spoof.py tidak ditemukan. Fitur verifikasi akan dinonaktifkan.")
    VERIFICATION_AVAILABLE = False
    run_complete_verification = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow - Multi-Face Recognition")
        self.setFixedSize(800, 600)
        
        # Inisialisasi window di __init__
        self.presensi_window = None
        self.registrasi_window = None
        self.training_window = None
        
        self.init_ui()
        self.init_dataset()
    
    def init_ui(self):
        """Initialize UI components - TIDAK DIUBAH dari desain asli"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title label - sesuai desain asli
        title_label = QLabel("MULTI-FACE RECOGNITION DAN GAZE TRACKING UNTUK AKSES PEGAWAI")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                margin: 20px 0;
            }
        """)
        layout.addWidget(title_label)
        layout.addSpacing(50)
        
        # Buttons - sesuai desain asli
        buttons = [
            ("PRESENSI", self.start_verification_process),  # Ubah approach ini
            ("REGISTRASI", self.show_registrasi),
            ("TRAIN", self.show_training),
            ("EXIT", self.close)
        ]
        
        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setFixedSize(200, 40)
            btn.clicked.connect(callback)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    border: 1px solid #999;
                    border-radius: 2px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
            layout.addWidget(btn, alignment=Qt.AlignCenter)
            layout.addSpacing(20)
        
        layout.addStretch()
        
        # Window styling - sesuai desain asli
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QFrame { background-color: white; }
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #999;
                border-radius: 2px;
                padding: 5px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
            QLabel { font-size: 12px; }
        """)
    
    def init_dataset(self):
        """Initialize dataset directories"""
        os.makedirs("split/train", exist_ok=True)
        os.makedirs("split/valid", exist_ok=True)
        os.makedirs("split/test", exist_ok=True)
    
    def start_verification_process(self):
        """Memulai proses verifikasi dengan subprocess untuk menghindari threading issue"""
        # Cek apakah requirements terpenuhi
        if not self.check_requirements():
            return
        
        # Disable tombol presensi
        self.set_presensi_button_state(False, "Verifikasi...")
        
        try:
            # Jalankan verifikasi dalam subprocess terpisah
            print("Memulai verifikasi dalam subprocess...")
            
            # Cek apakah file verification_subprocess.py ada
            if os.path.exists("verification_subprocess.py"):
                # Jalankan sebagai subprocess
                result = subprocess.run([sys.executable, "verification_subprocess.py"], 
                                      capture_output=True, text=True)
                success = (result.returncode == 0)
            else:
                # Fallback ke method lama
                if VERIFICATION_AVAILABLE and run_complete_verification:
                    success = run_complete_verification()
                else:
                    success = True  # Skip verifikasi jika tidak tersedia
            
            self.set_presensi_button_state(True, "PRESENSI")
            
            if success:
                # Verifikasi berhasil, buka window presensi
                QTimer.singleShot(500, self.show_presensi)
                QMessageBox.information(self, "Sukses", "Verifikasi berhasil! Membuka sistem presensi...")
            else:
                # Verifikasi gagal
                QMessageBox.warning(
                    self, 
                    "Verifikasi Gagal",
                    "Verifikasi anti-spoof gagal atau dibatalkan.\n\n"
                    "Kemungkinan penyebab:\n"
                    "• Wajah tidak terdaftar dalam sistem\n"
                    "• Perintah gaze tidak diikuti dengan benar\n"
                    "• Timeout verifikasi\n"
                    "• User membatalkan proses\n\n"
                    "Silakan coba lagi dan pastikan:\n"
                    "1. Wajah Anda sudah terdaftar\n"
                    "2. Ikuti semua perintah dengan benar\n"
                    "3. Posisi wajah stabil di depan kamera"
                )
        
        except Exception as e:
            self.set_presensi_button_state(True, "PRESENSI")
            QMessageBox.critical(self, "Error", f"Error dalam verifikasi: {str(e)}")
    
    def check_requirements(self):
        """Cek apakah semua file yang diperlukan tersedia"""
        missing_items = []
        
        # Cek model
        if not os.path.exists("model_final.keras"):
            missing_items.append("Model AI belum dilatih")
        
        # Cek dataset test
        if not os.path.exists("split/test") or not os.listdir("split/test"):
            missing_items.append("Belum ada wajah yang terdaftar")
        else:
            test_folders = [d for d in os.listdir("split/test") 
                           if os.path.isdir(os.path.join("split/test", d))]
            if not test_folders:
                missing_items.append("Belum ada wajah yang terdaftar")
        
        # Cek sistem verifikasi
        if not VERIFICATION_AVAILABLE:
            missing_items.append("Library gaze-tracking belum terinstall")
        
        if missing_items:
            message = "Requirements belum terpenuhi:\n\n"
            for i, item in enumerate(missing_items, 1):
                message += f"{i}. {item}\n"
            
            message += "\nSilakan lengkapi requirements terlebih dahulu:\n"
            message += "- Lakukan REGISTRASI untuk mendaftarkan wajah\n"
            message += "- Lakukan TRAINING untuk melatih model\n"
            message += "- Install library: pip install gaze-tracking"
            
            QMessageBox.warning(self, "Requirements Tidak Terpenuhi", message)
            return False
        
        return True
    
    def set_presensi_button_state(self, enabled, text="PRESENSI"):
        """Set state tombol presensi"""
        for child in self.centralWidget().findChildren(QPushButton):
            if "PRESENSI" in child.text() or "Verifikasi" in child.text():
                child.setEnabled(enabled)
                child.setText(text)
                break
    
    def show_presensi(self):
        """Show presensi window (dipanggil setelah verifikasi berhasil)"""
        try:
            if self.presensi_window is None:
                self.presensi_window = PresensiWindow()
            self.presensi_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Presensi: {str(e)}")
            self.presensi_window = None
    
    def show_registrasi(self):
        """Show registrasi window"""
        try:
            if self.registrasi_window is None:
                self.registrasi_window = RegistrasiWindow()
            self.registrasi_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Registrasi: {str(e)}")
            self.registrasi_window = None
    
    def show_training(self):
        """Show training window"""
        try:
            if self.training_window is None:
                self.training_window = TrainingWindow()
            self.training_window.show()
            
            # Check if training data exists
            if not os.listdir("split/train"):
                QMessageBox.warning(
                    self, 
                    "Empty Dataset", 
                    "No training data found! Please register faces first."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Training: {str(e)}")
            self.training_window = None
    
    def closeEvent(self, event):
        """Handle window close event"""
        event.accept()