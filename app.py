import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import dlib
from deepface import DeepFace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import distance as dist
import random
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from flask import jsonify

app = Flask(__name__)
CORS(app)

# print("hello from backend")


# Spotify setup
SPOTIFY_CLIENT_ID = "19766267968046b695c249d3b3015b9f"
SPOTIFY_CLIENT_SECRET = "6a89aa34a70d4f43972b5ab717ac461e"

spotify = Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET
))

# Dlib face detector and shape predictor for drowsiness detection
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye Aspect Ratio (EAR) threshold for drowsiness
EAR_THRESHOLD = 0.12
MAR_THRESHOLD = 0.6 
DROWSINESS_FRAMES = 5 # Number of consecutive frames for drowsiness detection

# Directory to save frames

# Function to extract frames
def extract_frames(video_path, skip_frames=10):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:

            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames

# Function to detect faces using dlib
def detect_faces(frames):
    face_crops = []
    face_locations = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dlib_detector(gray)
        
        if len(faces) == 0:
            print("No faces detected in frame.")

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_crops.append(frame[y:y+h, x:x+w])
            face_locations.append((x, y, w, h))
            
    print(f"Detected faces: {len(face_crops)}")

    return face_crops, face_locations

# Function to compute Eye Aspect Ratio (EAR)
def compute_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to compute Mouth Aspect Ratio (MAR)
def compute_mar(mouth):
    A = dist.euclidean(mouth[13], mouth[19])  # Vertical distance (mouth open)
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])  # Horizontal mouth width
    mar = (A + B + C) / (3.0 * D)
    return mar

# Function to detect drowsiness
def detect_drowsiness(frames, face_locations):
    drowsy_count = 0
    yawn_count=0

    for frame, faces in zip(frames, face_locations):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dlib_detector(gray)

        for face in faces:
            shape = dlib_predictor(gray, face)
            left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
            right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
            
            # Mouth
            mouth = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            mar = compute_mar(mouth)

            if avg_ear < EAR_THRESHOLD:
                drowsy_count += 1
                
            if mar > MAR_THRESHOLD:
                yawn_count += 1

    # If drowsy or yawning detected over multiple frames
    return (drowsy_count >= DROWSINESS_FRAMES) or (yawn_count >= DROWSINESS_FRAMES)

# Parallel Emotion Recognition
def recognize_emotions_parallel(faces):
    emotions = []
    print(f"Recognizing emotions for {len(faces)} faces.")

    def analyze_face(face):
        try:
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            return analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        results = executor.map(analyze_face, faces)

    emotions = [res for res in results if res] 
    print(f"Emotion analysis result: {emotions}")
    return emotions

# Function to get music recommendations
def get_music_recommendations(dominant_emotions, language, preference=""):
    emotion_queries = {
        "happy": ["happy", "upbeat", "energetic"],
        "sad": ["sad", "melancholic", "emotional"],
        "angry": ["intense", "powerful", "rock"],
        "surprise": ["trending", "fresh", "exciting"],
        "fear": ["calm", "soothing", "ambient"],
        "neutral": ["chill", "relaxing", "lofi"]
    }
    
    language_queries = {
        "hindi": "bollywood",
        "english": "pop",
        "punjabi": "punjabi",
        "tamil": "tamil hits"
    }
    
    recommendations = []
    lang_query = language_queries.get(language, "bollywood")
    
    for emotion, _ in dominant_emotions:
        search_queries = emotion_queries.get(emotion, ["top"])
        
        # If user preference is given, use it as priority
        if preference:
            search_queries.insert(0, preference)
            
        query = f"{random.choice(search_queries)} {lang_query}"

        results = spotify.search(q=query, type="track", limit=15)
        
        if "tracks" in results and "items" in results["tracks"]:
            popular_tracks = results["tracks"]["items"]
            random.shuffle(popular_tracks)
            
            for track in popular_tracks[:3]:
                recommendations.append({
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "url": track["external_urls"]["spotify"],
                    "popularity": track["popularity"]
                })
    
    return recommendations

@app.route('/')
def index():
    return "Flask is working!"

@app.route('/process_video', methods=['POST'])
def process_video():
    
    print("hello")
    if 'video' not in request.files:
        return jsonify({"error": "No video file found"}), 400

    video = request.files['video']
    
    print("Received video file:", video.filename)  # Log the filename to verify the upload
     
    video_path = 'uploaded_video.mp4'
    video.save(video_path)
    
    print(video)
    
    user_preference = request.form.get("preference", "").lower()


    frames = extract_frames(video_path, skip_frames=5)
    faces, face_locations = detect_faces(frames)
    emotions = recognize_emotions_parallel(faces)
    
    emotion_counts = Counter(emotions)
    if "neutral" in emotion_counts:
        del emotion_counts["neutral"]
    
    all_emotions = emotion_counts.most_common()
    user_language = request.form.get("language", "hindi").lower()
    music_recommendations = get_music_recommendations(all_emotions, user_language,user_preference)

    # Detect Drowsiness
    is_drowsy = detect_drowsiness(frames, face_locations)
    
    if is_drowsy:
        music_recommendations = get_music_recommendations([("happy", 1)], user_language,user_preference)  # Energetic tracks

    return jsonify({
        'emotion_counts': all_emotions,
        'music_recommendations': music_recommendations,
        'drowsiness_detected': is_drowsy
    })

if __name__ == "__main__":
     app.run(debug=True, host='0.0.0.0', port=5000)
