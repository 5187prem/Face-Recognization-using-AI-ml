import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime

# Load known images
path = 'images'
images = []
classNames = []

print("🔄 Loading known images...")
if not os.path.exists(path):
    print("❌ 'images/' folder not found.")
    exit()

myList = os.listdir(path)
if not myList:
    print("❌ No images found in 'images/' folder.")
    exit()

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        name = os.path.splitext(cl)[0].rsplit('_', 1)[0]
        classNames.append(name)
print(f"✅ Loaded images: {classNames}")

# Encode known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

encodeListKnown = findEncodings(images)
print("✅ Encodings complete.")

# Attendance tracking
attendance_marked = set()

def markAttendance(name):
    filename = 'Attendance.csv'
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Avoid duplicate entry in the same session
        if name in attendance_marked:
            return

        # Check if already marked today
        already_marked = False
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if name in line and date_str in line:
                        already_marked = True
                        break

        if not already_marked:
            with open(filename, 'a') as f:
                f.write(f'{name},{timestamp}\n')
            print(f"📝 Attendance marked: {name} at {timestamp}")
            attendance_marked.add(name)
        else:
            print(f"ℹ️ {name} already marked today.")
    except Exception as e:
        print(f"❌ Error writing to attendance: {e}")

# Webcam setup
cap = cv2.VideoCapture(0)
print("📷 Webcam started.")

start_time = time.time()
tolerance = 0.4  # Lower = more strict

while True:
    success, img = cap.read()
    if not success:
        print("❌ Webcam failed.")
        break

    if time.time() - start_time > 25:
        print("⏹️ Time's up. Exiting.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgRGB)
    encodesCurFrame = face_recognition.face_encodings(imgRGB, facesCurFrame)

    print(f"🧠 Faces detected: {len(encodesCurFrame)}")

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) == 0:
            continue

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].capitalize()
            confidence = 1 - faceDis[matchIndex]
            label = f'{name} ({confidence:.2f})'

            print(f"✅ Match: {label}")
            markAttendance(name)

            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == 27:  # ESC to quit
        print("🛑 ESC pressed.")
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited cleanly.")
