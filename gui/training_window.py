from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QGridLayout, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1_l2

class TrainingThread(QThread):
    update_signal = pyqtSignal(str, float, float, float, float, object)

    def run(self):
        try:
            print("[INFO] Loading pre-trained model and dataset...")
            
            # PERBAIKAN 1: Gunakan folder terpisah untuk train dan validation
            train_path = "split/train"
            val_path = "split/valid"  # Pastikan folder ini ada!
            
            # Cek apakah folder validation ada
            if not os.path.exists(val_path):
                self.update_signal.emit("Error: Folder 'split/valid' tidak ditemukan! Pastikan dataset sudah dibagi dengan benar.", 0, 0, 0, 0, None)
                return

            class_count = len(next(os.walk(train_path))[1])
            
            if class_count == 0:
                self.update_signal.emit("Dataset Kosong! Tidak ada kelas ditemukan.", 0, 0, 0, 0, None)
                return

            image_size = (224, 224)
            batch_size = 16  # PERBAIKAN 2: Batch size lebih kecil untuk stabilitas

            # PERBAIKAN 3: Augmentasi HANYA untuk training data
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=15,        # Kurangi rotasi
                width_shift_range=0.1,    # Kurangi shift
                height_shift_range=0.1,
                shear_range=0.1,          # Kurangi shear
                zoom_range=0.1,           # Kurangi zoom
                horizontal_flip=True,
                brightness_range=[0.9, 1.1],  # Variasi brightness ringan
                fill_mode='nearest'
            )

            # PERBAIKAN 4: Validation tanpa augmentasi
            val_datagen = ImageDataGenerator(rescale=1.0 / 255)

            # Generator terpisah untuk train dan validation
            train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=image_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True
            )

            val_generator = val_datagen.flow_from_directory(
                val_path,
                target_size=image_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False  # Validation tidak perlu shuffle
            )

            class_count = len(train_generator.class_indices)

            # Load VGG16
            base_model = VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=(224, 224, 3)
            )

            base_model.trainable = False

            # PERBAIKAN 5: Model dengan regularisasi yang lebih kuat
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                
                # Layer dengan regularisasi L1+L2
                Dense(128, 
                      activation="relu",
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
                BatchNormalization(),
                Dropout(0.7),  # Dropout lebih tinggi
                
                Dense(64, 
                      activation="relu",
                      kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
                BatchNormalization(),
                Dropout(0.5),
                
                Dense(class_count, activation="softmax")
            ])

            # PERBAIKAN 6: Learning rate yang lebih sesuai
            model.compile(
                optimizer=Adam(learning_rate=0.0005),  # Lebih kecil dari default
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            # PERBAIKAN 7: Callbacks yang lebih ketat
            callbacks_phase1 = [
                EarlyStopping(
                    monitor="val_loss", 
                    patience=5, 
                    restore_best_weights=True,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-7
                ),
                ModelCheckpoint(
                    "model_phase1.keras", 
                    save_best_only=True, 
                    monitor="val_loss",
                    mode='min'
                )
            ]

            # Training Phase 1: Transfer Learning
            print("[INFO] Starting Phase 1: Transfer Learning...")
            history1 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=15,  # Kurangi epoch
                callbacks=callbacks_phase1,
                verbose=1
            )

            # PERBAIKAN 8: Fine-tuning yang lebih konservatif
            base_model.trainable = True

            # Freeze lebih banyak layer (hanya unfreeze 4 layer terakhir)
            for layer in base_model.layers[:-4]:
                layer.trainable = False

            # PERBAIKAN 9: Learning rate yang tepat untuk fine-tuning
            model.compile(
                optimizer=Adam(learning_rate=1e-5),  # Sangat kecil untuk fine-tuning
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            callbacks_phase2 = [
                EarlyStopping(
                    monitor="val_loss", 
                    patience=3,  # Patience lebih kecil
                    restore_best_weights=True,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-8
                ),
                ModelCheckpoint(
                    "model_finetuned.keras", 
                    save_best_only=True, 
                    monitor="val_loss",
                    mode='min'
                )
            ]

            # Training Phase 2: Fine-tuning
            print("[INFO] Starting Phase 2: Fine-tuning...")
            history2 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=10,  # Epoch lebih sedikit untuk fine-tuning
                callbacks=callbacks_phase2,
                verbose=1
            )

            model.save("model_final.keras")
            print("Model telah disimpan sebagai 'model_final.keras'.")
            
            # Combine histories
            combined_history = {
                'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                'loss': history1.history['loss'] + history2.history['loss'],
                'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
                'val_loss': history1.history['val_loss'] + history2.history['val_loss']
            }
            
            # Final results
            final_train_acc = history2.history['accuracy'][-1]
            final_train_loss = history2.history['loss'][-1]
            final_val_acc = history2.history['val_accuracy'][-1]
            final_val_loss = history2.history['val_loss'][-1]

            # PERBAIKAN 10: Peringatan jika terjadi overfitting
            accuracy_gap = final_train_acc - final_val_acc
            if accuracy_gap > 0.1:  # Gap lebih dari 10%
                message = f"Training Selesai! PERINGATAN: Kemungkinan overfitting (Gap accuracy: {accuracy_gap:.2%})"
            else:
                message = "Training Selesai! Model terlihat stabil."

            self.update_signal.emit(
                message,
                final_train_acc, 
                final_train_loss, 
                final_val_acc, 
                final_val_loss,
                combined_history
            )
        
        except Exception as e:
            self.update_signal.emit(f"Error: {str(e)}", 0, 0, 0, 0, None)

# Class TrainingWindow tetap sama, hanya tambahkan informasi tentang dataset structure
class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pre-Trained Model Training - Fixed Overfitting")
        self.setFixedSize(900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.title_label = QLabel("VGG16 Transfer Learning - Anti-Overfitting Version")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2E86AB;")
        main_layout.addWidget(self.title_label)

        # PERBAIKAN 11: Informasi tentang struktur dataset yang benar
        description = QLabel(
            "PENTING: Pastikan struktur dataset Anda:\n"
            "split/\n"
            "  ├── train/\n"
            "  │   ├── kelas1/\n"
            "  │   └── kelas2/\n"
            "  ├── valid/\n"
            "  │   ├── kelas1/\n"
            "  │   └── kelas2/\n"
            "  └── test/\n"
            "      ├── kelas1/\n"
            "      └── kelas2/\n\n"
            "Model ini menggunakan regularisasi dan callbacks yang ketat untuk mencegah overfitting."
        )
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 11px; margin: 10px; background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        main_layout.addWidget(description)

        # Grafik (sama seperti sebelumnya)
        self.figure1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvas(self.figure1)
        self.ax1.set_title('Training & Validation Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')

        self.figure2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvas(self.figure2)
        self.ax2.set_title('Training & Validation Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')

        plots_layout = QHBoxLayout()
        plots_layout.addWidget(self.canvas1)
        plots_layout.addWidget(self.canvas2)
        main_layout.addLayout(plots_layout)

        # Metrics layout (sama seperti sebelumnya)
        metrics_layout = QGridLayout()
        
        metrics_layout.addWidget(QLabel("Train Dataset:"), 0, 0)
        self.dataset_label = QLabel("split/train")
        metrics_layout.addWidget(self.dataset_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Validation Dataset:"), 1, 0)
        self.val_dataset_label = QLabel("split/valid")
        self.val_dataset_label.setStyleSheet("color: #D2691E; font-weight: bold;")
        metrics_layout.addWidget(self.val_dataset_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Model:"), 2, 0)
        self.model_label = QLabel("VGG16 + Regularization")
        metrics_layout.addWidget(self.model_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("Training Accuracy:"), 3, 0)
        self.train_acc_label = QLabel("-")
        metrics_layout.addWidget(self.train_acc_label, 3, 1)

        metrics_layout.addWidget(QLabel("Training Loss:"), 4, 0)
        self.train_loss_label = QLabel("-")
        metrics_layout.addWidget(self.train_loss_label, 4, 1)

        metrics_layout.addWidget(QLabel("Validation Accuracy:"), 5, 0)
        self.val_acc_label = QLabel("-")
        metrics_layout.addWidget(self.val_acc_label, 5, 1)

        metrics_layout.addWidget(QLabel("Validation Loss:"), 6, 0)
        self.val_loss_label = QLabel("-")
        metrics_layout.addWidget(self.val_loss_label, 6, 1)
        
        # Tambahkan indikator overfitting
        metrics_layout.addWidget(QLabel("Status Model:"), 7, 0)
        self.overfitting_label = QLabel("Belum dilatih")
        metrics_layout.addWidget(self.overfitting_label, 7, 1)

        main_layout.addLayout(metrics_layout)

        self.train_button = QPushButton("Mulai Training (Anti-Overfitting)")
        self.train_button.setStyleSheet("QPushButton { background-color: #2E86AB; color: white; font-weight: bold; padding: 10px; }")
        self.train_button.clicked.connect(self.start_training)
        main_layout.addWidget(self.train_button)

        self.status_label = QLabel("Status: Menunggu Training...")
        main_layout.addWidget(self.status_label)

    def start_training(self):
        self.status_label.setText("Status: Training dengan Anti-Overfitting measures...")
        self.train_button.setEnabled(False)

        self.training_thread = TrainingThread()
        self.training_thread.update_signal.connect(self.update_results)
        self.training_thread.start()

    def update_results(self, message, train_acc, train_loss, val_acc, val_loss, history=None):
        self.status_label.setText(f"Status: {message}")
        self.train_acc_label.setText(f"{train_acc:.2%}")
        self.train_loss_label.setText(f"{train_loss:.4f}")
        self.val_acc_label.setText(f"{val_acc:.2%}")
        self.val_loss_label.setText(f"{val_loss:.4f}")
        self.train_button.setEnabled(True)
        
        # Update overfitting indicator
        accuracy_gap = train_acc - val_acc
        if accuracy_gap < 0.05:  # Gap kurang dari 5%
            self.overfitting_label.setText("✓ Model Stabil")
            self.overfitting_label.setStyleSheet("color: green; font-weight: bold;")
        elif accuracy_gap < 0.15:  # Gap 5-15%
            self.overfitting_label.setText("⚠ Sedikit Overfitting")
            self.overfitting_label.setStyleSheet("color: orange; font-weight: bold;")
        else:  # Gap > 15%
            self.overfitting_label.setText("❌ Overfitting Parah")
            self.overfitting_label.setStyleSheet("color: red; font-weight: bold;")
        
        # Update plots
        if history is not None:
            self.ax1.clear()
            self.ax2.clear()
            
            self.ax1.set_title('Training & Validation Loss')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            
            self.ax2.set_title('Training & Validation Accuracy')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Accuracy')
            
            epochs = range(1, len(history['accuracy']) + 1)
            
            # Plot with different colors for better distinction
            self.ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
            self.ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            
            self.ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            self.ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            # Add vertical line to show where fine-tuning started (assuming phase 1 was 15 epochs)
            if len(epochs) > 15:
                self.ax1.axvline(x=15, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning start')
                self.ax2.axvline(x=15, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning start')
            
            self.canvas1.draw()
            self.canvas2.draw()

# Untuk menjalankan aplikasi
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())