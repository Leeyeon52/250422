import os

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

SEED = 2021



def load_cifar10_dataset():
    train_X = np.load("cifar10_train_X.npy").astype(np.float32)
    train_y = np.load("cifar10_train_y.npy").astype(np.int32)  # 정수형으로 변환
    test_X = np.load("cifar10_test_X.npy").astype(np.float32)
    test_y = np.load("cifar10_test_y.npy").astype(np.int32)
    
    print("train_X dtype:", train_X.dtype)
    print("train_y dtype:", train_y.dtype)
    print("Max value in train_X:", train_X.max())
    print("Min value in train_X:", train_X.min())
    print("Max value in train_y:", train_y.max())
    print("Min value in train_y:", train_y.min())
    
    train_X, test_X = train_X / 255.0, test_X / 255.0
    
    return train_X, train_y, test_X, test_y

def build_mlp_model(img_shape, num_classes=10):
    model = Sequential()
    model.add(Input(shape=img_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation="relu"))
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model

# EarlyStopping 콜백함수의 인스턴스를 cb_earlystop에 저장합니다.
# TODO: 지시사항 1을 참고하여 매개변수를 설정하세요
cb_earlystop = tf.keras.callbacks.EarlyStopping( monitor= 'val_loss' , 
                                                 mode= 'auto',
                                                 verbose=1,
                                                 patience=2 )
                                                 
# ModelCheckpoint 콜백함수의 인스턴스를 cb_chkpnt 저장합니다.
# TODO: 지시사항 2을 참고하여 매개변수를 설정하세요
cb_chkpnt = tf.keras.callbacks.ModelCheckpoint( filepath = "./chkpnt/{epoch:04d}.ckpt",  
                                             monitor='val_loss',
                                             mode='auto',  
                                             verbose=1, 
                                             save_best_only = True,
                                             save_weights_only =False,
                                             save_freq='epoch' )

def main(epochs=10):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    
    train_X, train_y, test_X, test_y = load_cifar10_dataset()
    img_shape = train_X[0].shape

    print(img_shape)

    optimizer = Adam(learning_rate=1e-3)

    mlp_model = build_mlp_model(img_shape)
    
    mlp_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    hist = mlp_model.fit(train_X, train_y, epochs=epochs, batch_size=32, validation_split=0.2, shuffle=True, verbose=1, callbacks=[cb_earlystop, cb_chkpnt])
    # TODO: cb_earlystop와 cb_chkpnt를 리스트로 묶어 fit 함수의 callbacks 매개변수로 전달하세요.

    return optimizer, hist

if __name__ == "__main__":
    main()