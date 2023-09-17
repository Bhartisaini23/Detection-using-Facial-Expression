import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam, SGD

   frontalface_detector = dlib.get_frontal_face_detector()
   def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
   def detect_face(image_url):
    try:
        url_response = urllib.request.urlopen(image_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        rects = frontalface_detector(image, 1)
   if len(rects) < 1:
        return "No Face Detected"
   for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(image, interpolation='nearest')
        plt.axis('off')
        plt.show()

   frontalface_detector = dlib.get_frontal_face_detector()
   landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


   def get_landmarks(image_url):
    try:
        url_response = urllib.request.urlopen(image_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
    except Exception as e:
        print("Please check the URL and try again!")
        return None, None
    faces = frontalface_detector(image, 1)
      if len(faces):
        landmarks = [(p.x, p.y) for p in landmark_predictor(image, faces[0]).parts()]
      else:
        return None, None

        return image, landmarks


    def image_landmarks(image, face_landmarks):
    radius = -1
    circle_thickness = 4
    image_copy = image.copy()
    for (x, y) in face_landmarks:
        cv2.circle(image_copy, (x, y), circle_thickness, (255, 0, 0), radius)
        plt.imshow(image_copy, interpolation='nearest')
        plt.axis('off')
        plt.show()

        #In this model, the specific landmarks for facial features are:
        #Jawline = 0–16 Eyebrows = 17–26 Nose = 27–35 Left eye = 36–41 Right eye = 42–47Mouth = 48–67
        #The target labels are integer encoded in the csvfile.They are mapped as follows:
        #0 — -> Angry  1 — -> Happy  2 — -> Sad 3 — -> Surprise 4 — -> Neutral

        label_map = {"0": "ANGRY", "1": "HAPPY", "2": "SAD", "3": "SURPRISE", "4": "NEUTRAL"}
        df = pd.read_csv("./ferdata.csv")
        df.head()

       #Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.1, random_state=42,stratify=dataY)

    #Standardise the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_test)

    #To find the accuracy of the model:
    print(accuracy_score(Y_test, predictions)*100)

    mlp_model = Sequential()
    mlp_model.add(Dense(1024, input_shape = (2304,), activation = 'relu', kernel_initializer='glorot_normal'))
    mlp_model.add(Dense(512, activation = 'relu', kernel_initializer='glorot_normal'))
    mlp_model.add(Dense(5, activation = 'softmax'))
    mlp_model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=0.001), metrics=['accuracy'])
    checkpoint = ModelCheckpoint('best_mlp_model.h5',verbose=1,monitor='val_acc', save_best_only=True,mode='auto')
    mlp_history = mlp_model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data(X_test, y_test),shuffle=True)

    width, height = 48, 48
    X_train = X_train.reshape(len(X_train),height,width)
    X_test = X_test.reshape(len(X_test),height,width)
    X_train = np.expand_dims(X_train,3)
    X_test = np.expand_dims(X_test,3)
    cnn_model = Sequential()
    cnn_model.add(Conv2D(5000, kernel_size=(4, 4), activation='relu', padding='same', input_shape = (width, height, 1)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(3, 3), strides=(4, 4)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(2000, activation='relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(5, activation='softmax'))
    checkpoint = ModelCheckpoint('best_cnn_model.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    cnn_model.compile(loss=categorical_crossentropy,
    optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
    metrics=['accuracy'])
    cnn_history = cnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint],
    validation_data=(X_test, y_test),shuffle=True)
    cnn_model.save('cnn_model.h5')