from flask import Flask, render_template, request, redirect, url_for
import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime

app = Flask(__name__)

# Load trained face encodings
known_faces = np.load('known_faces.npy', allow_pickle=True)
known_names = np.load('known_names.npy', allow_pickle=True)

# Create folder to save captured images (optional)
if not os.path.exists('static/captured'):
    os.makedirs('static/captured')

# Attendance file
attendance_file = 'attendance.csv'

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

    # If file not exists, create it with header
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Time\n')

    # Check if already marked today
    # Use 'a+' to read and append (so we don't overwrite)
    with open(attendance_file, 'r+') as f:
        data = f.readlines()
        names_today = []
        for line in data[1:]: # Skip header
            entry = line.split(',')
            if len(entry) >= 2:
                # Check if entry date is today
                if entry[1].startswith(now.strftime('%Y-%m-%d')):
                    names_today.append(entry[0])

        if name not in names_today:
            # Move pointer to the end of the file to append
            f.seek(0, os.SEEK_END)
            f.write(f'{name},{dt_string}\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    cap = cv2.VideoCapture(0)
    
    # Give camera a moment to initialize
    if not cap.isOpened():
        return "Camera Error: Could not open webcam."
        
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Camera Error: Could not read frame."

    # Save captured image
    img_path = os.path.join('static/captured', 'captured.jpg')
    cv2.imwrite(img_path, frame)

    # --- THIS IS THE FIX ---
    # Recognize face
    # rgb_frame = frame[:, :, ::-1] # OLD line, causes TypeError
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # NEW line, fixes TypeError
    # --- END OF FIX ---
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    name = "Unknown"
    
    if len(face_encodings) > 0:
        # Use only the first face found
        face_encoding = face_encodings[0]
        
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

    # Mark attendance
    if name != "Unknown":
        mark_attendance(name)

    # Pass a random string to the image path to force browser to refresh the cache
    import random
    cache_buster = random.randint(1, 10000)
    
    return render_template('index.html', name=name, image_path=f"{img_path}?v={cache_buster}")

if __name__ == '__main__':
    app.run(debug=True)