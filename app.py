import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import spotipy
from spotipy.oauth2 import SpotifyOAuth

model = load_model('face_model.h5')

# Define emotion categories and related playlists
emotion_playlists = {
    'Angry': 'spotify:playlist: ---- Playlist link goes here------------',  # Rock Playlist
    'Surprised': 'spotify:playlist:---- Playlist link goes here------------',  # Pop Playlist
    'Neutral': 'spotify:playlist:---- Playlist link goes here------------',  # Classical Playlist
    'Sad': 'spotify:playlist:---- Playlist link goes here------------',  # Blues Playlist
    'Happy': 'spotify:playlist:---- Playlist link goes here------------'  # Electronic Playlist
}

# Initialize Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="--------put your client id ---------",
                                                client_secret="--------put your secret key id ---------",
                                                redirect_uri="http://localhost:8888/callback",
                                                scope="user-read-playback-state,user-modify-playback-state"))

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Track the current emotion to avoid restarting playback
current_emotion = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        predictions = model.predict(roi_gray)
        max_index = np.argmax(predictions[0])
        emotion = list(emotion_playlists.keys())[max_index]

        # Display emotion on frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Play related playlist only if emotion changes
        if emotion != current_emotion:
            playlist_uri = emotion_playlists[emotion]
            try:
                sp.start_playback(context_uri=playlist_uri)
                current_emotion = emotion  # Update the current emotion
            except Exception as e:
                print(f"Spotify Error: {e}")

    cv2.imshow('Mood Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()