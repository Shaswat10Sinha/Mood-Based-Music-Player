# Mood Based Music Player

## Overview
This application detects emotions in real time using facial expressions and plays music matching the detected mood through Spotify playlists.





![flowchart drawio](https://github.com/user-attachments/assets/a0217c0f-183c-49e8-bdf1-d2eb2be25518)

## Key Features:

Detects emotions (happy, sad, angry, etc.) using a CNN model trained on the FER2013 dataset.

Plays Spotify playlists based on detected moods.


## Installation Instructions:

### Clone the repository:

git clone https://github.com/Shaswat10Sinha/Mood-Based-Music-Player.git

### Install dependencies:

pip install -r requirements.txt

Download FER2013 Dataset and place it in the 'data/' folder.

### Train the CNN Model:

facial expression.py

Save the model as an H5 file for deployment.

### Run the application:

python app.py

### Spotify API Setup:

Register your application on Spotify Developer Dashboard.

Note down your Client ID and Client Secret.

Update the .env file with the credentials.

### Usage:

Launch the UI and grant webcam access.

Start the emotion detection process.

Enjoy the auto-curated playlists based on your mood.

## Tech Stack:

Machine Learning: TensorFlow, Keras

Computer Vision: OpenCV

Music Integration: Spotify API

UI Development: Gradio/Streamlit

## Future Improvements:

Add support for more emotions.

Customize playlists.

Extend the UI for public use.

Add multi-user support.
