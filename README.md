# EmotionAnalysisProject
 
# Emotion Analysis Project

This project uses computer vision to classify facial expressions into seven emotion categories:
- 0 = Angry
- 1 = Disgust
- 2 = Fear
- 3 = Happy
- 4 = Sad
- 5 = Surprise
- 6 = Neutral

The dataset consists of 48x48 pixel grayscale images of faces. A Streamlit web interface allows users to upload an image and predict the emotion shown in the facial expression.

## Features
- Emotion classification using a Convolutional Neural Network (CNN).
- Streamlit-based web app for image upload and prediction.

## Project Structure
```
├── dataset/                # Directory containing images organized by emotion labels
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── surprise/
│   │   └── neutral/
│   └── test/               # Similar structure to train/
├── emotion_model.h5        # Saved trained CNN model
├── app.py                  # Streamlit app for prediction
└── README.md               # Project documentation
```

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.x installed, then install the required libraries:
```bash
pip install numpy pandas tensorflow keras opencv-python streamlit
```

### 2. Prepare the Dataset
Unzip the dataset and organize it into `train` and `test` directories with subfolders for each emotion category.

Example structure:
```
dataset/train/angry
             /disgust
             /fear
             /happy
             /sad
             /surprise
             /neutral
```

### 3. Train the Model


## License
This project is open-source and available for educational use.

