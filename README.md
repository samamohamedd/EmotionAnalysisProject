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
Run the following script to train the CNN model:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'path_to_extracted/train', target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'path_to_extracted/test', target_size=(48, 48), color_mode='grayscale', batch_size=64, class_mode='categorical')

# Model definition
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=test_generator, epochs=20)

model.save('emotion_model.h5')
```

### 4. Streamlit Web App
Create `app.py` to allow users to upload an image and predict emotions.

```python
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Emotion Analysis from Facial Expressions")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img_resized = image.resize((48, 48))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)
    
    prediction = model.predict(img_array)
    emotion = emotion_labels[np.argmax(prediction)]
    
    st.image(uploaded_file, caption=f"Predicted Emotion: {emotion}")
```

### 5. Run the App
Use the following command to run the Streamlit app:
```bash
streamlit run app.py
```

## Future Enhancements
- Add data augmentation for better generalization.
- Improve the UI with additional styling and options.

## License
This project is open-source and available for educational use.

