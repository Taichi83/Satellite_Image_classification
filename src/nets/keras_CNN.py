from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model

def Keras_CNN(input_shape, classes, DROPOUT=0.5):
    img_input = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Dropout(DROPOUT, name='block1_dropout')(x)

    # Block 2
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Dropout(DROPOUT, name='block2_dropout')(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Dropout(DROPOUT, name='block3_dropout')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(DROPOUT, name='dropout')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(img_input, x, name='cnn')
    print(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model