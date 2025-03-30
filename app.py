from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from deepface import DeepFace
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load movie dataset
df = pd.read_csv("./hollywood.csv")

# Ensure 'rating' column exists
if "rating" not in df.columns:
    df["rating"] = 5.0  # Default rating if missing

# Mood-to-genre mapping
mood_to_genre = {
    "happy": ["Comedy", "Romance"],
    "sad": ["Drama", "Documentary"],
    "angry": ["Action", "Thriller"],
    "surprise": ["Adventure", "Fantasy"],
    "fear": ["Horror", "Thriller"],
    "neutral": ["Drama", "Mystery"],
    "disgust": ["Horror", "Crime"]
}

# Weather-to-genre mapping
weather_to_genre = {
    "Clear": ["Adventure", "Comedy", "Sci-Fi"],
    "Rain": ["Drama", "Romance", "Mystery"],
    "Drizzle": ["Drama", "Romance", "Mystery"],
    "Snow": ["Animation", "Fantasy", "Family"],
    "Fog": ["Thriller", "Horror", "Mystery"],
    "Mist": ["Thriller", "Horror", "Mystery"],
    "Thunderstorm": ["Action", "Horror", "Sci-Fi"],
    "Wind": ["Fantasy", "Action", "Western"],
    "Clouds": ["Drama", "Mystery", "Documentary"]
}

def analyze_emotion(image):
    """Detect mood from image using DeepFace."""
    try:
        img_data = base64.b64decode(image)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list) and len(result) > 0:
            return result[0]['dominant_emotion']
        else:
            return "neutral"  # Default if no face detected
    
    except Exception as e:
        return "neutral"  # Return neutral if error occurs

def recommend_movies(genres):
    """Get top-rated movies based on genres."""
    if not genres:
        return []
    
    filtered_movies = df[df["genres"].str.contains("|".join(genres), case=False, na=False)]

    # Sort movies by rating if available, otherwise shuffle
    if "rating" in filtered_movies.columns:
        recommended = filtered_movies.sort_values(by="rating", ascending=False).head(5)
    else:
        recommended = filtered_movies.sample(n=min(5, len(filtered_movies)), random_state=42)

    return recommended[["title", "genres", "rating"]].to_dict(orient="records")

@app.route('/detect_mood', methods=['POST'])
def detect_mood():
    """Detect mood from image and recommend movies."""
    data = request.json
    image = data.get('image')
    weather_condition = data.get('weather', 'Clear')  # Default to Clear

    if not image:
        return jsonify({"error": "No image received"}), 400

    mood = analyze_emotion(image)

    # Get movies by mood and weather
    mood_movies = recommend_movies(mood_to_genre.get(mood, ["Drama"]))
    weather_movies = recommend_movies(weather_to_genre.get(weather_condition, ["Drama"]))

    # Merge and ensure unique recommendations
    all_movies = {movie["title"]: movie for movie in (mood_movies + weather_movies)}

    return jsonify({
        "mood": mood,
        "weather_movies": weather_movies,
        "mood_movies": mood_movies,
        "combined_movies": list(all_movies.values())
    })

if __name__ == '__main__':
    app.run(debug=True)