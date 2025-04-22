import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 경로 설정
base_dir = './extracted_open_1'
train_img_dir = os.path.join(base_dir, 'train')
test_img_dir = os.path.join(base_dir, 'test')

# CSV 파일 불러오기
train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
artists_info_df = pd.read_csv('./extracted_open_1/artists_info.csv')
print(artists_info_df.columns)

# 클래스 이름을 숫자로 매핑
label_to_index = {name: idx for idx, name in enumerate(artists_info_df['artist'])}
train_df['label'] = train_df['artist'].map(label_to_index)

# 이미지 경로 추가
train_df['path'] = train_df['img_path'].apply(lambda x: os.path.join(train_img_dir, x))

# Train/Validation split
train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

# ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_dataframe(
    train_data, x_col='path', y_col='label', target_size=(224, 224), class_mode='raw', batch_size=32
)
val_generator = val_gen.flow_from_dataframe(
    val_data, x_col='path', y_col='label', target_size=(224, 224), class_mode='raw', batch_size=32
)

# 간단한 모델
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_to_index), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 학습
model.fit(train_generator, validation_data=val_generator, epochs=10)

# 저장
model.save('artist_classifier.h5')
