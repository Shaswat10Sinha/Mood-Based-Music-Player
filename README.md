# Mood Based Music Player

## Overview
This application detects emotions in real time using facial expressions and plays music matching the detected mood through Spotify playlists.

flowchart.drawio.png

## Key Features:

Detects emotions (happy, sad, angry, etc.) using a CNN model trained on the FER2013 dataset.

Plays Spotify playlists based on detected moods.


## Installation Instructions:

Clone the repository:

git clone 
cd mood-music-ai

Install dependencies:

pip install -r requirements.txt

Download FER2013 Dataset and place it in the 'data/' folder.

Train the CNN Model:

python train_model.py

Save the model as an H5 file for deployment.

Run the application:

python app.py

Spotify API Setup:

Register your application on Spotify Developer Dashboard.

Note down your Client ID and Client Secret.

Update the .env file with the credentials.

Usage:

Launch the UI and grant webcam access.

Start the emotion detection process.

Enjoy the auto-curated playlists based on your mood.

Tech Stack:

Machine Learning: TensorFlow, Keras

Computer Vision: OpenCV

Music Integration: Spotify API

UI Development: Gradio/Streamlit

Future Improvements:

Add support for more emotions.

Customize playlists.

Extend the UI for public use.

Add multi-user support.
