import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    # Ambil frame baru dari webcam
    _, frame = webcam.read()

    # Kirim frame ini ke GazeTracking untuk dianalisis
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Berkedip"
    elif gaze.is_right():
        text = "Lihat Kanan"
    elif gaze.is_left():
        text = "Lihat Kiri"
    elif gaze.is_center():
        text = "Lihat Tengah"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    # cv2.putText(frame, "Pupil Kiri:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.putText(frame, "Pupil Kanan: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Hitung rasio lebar mata terhadap tingginya
    if gaze.horizontal_ratio() is not None and gaze.vertical_ratio() is not None:
        horizontal_ratio = gaze.horizontal_ratio()
        vertical_ratio = gaze.vertical_ratio()
        #cv2.putText(frame, f"Rasio Lebar: {horizontal_ratio:.2f}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        #cv2.putText(frame, f"Rasio Tinggi: {vertical_ratio:.2f}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Deteksi wajah dan Arah Pandang", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
