#%%
from operator import truediv
from turtle import circle, color
import numpy as np
import mediapipe as mp
from mediapineline import*
# %%
mp_holistic= mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
#%%


# %% SETUP FOLDERS FOR COLLECTION: 
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
'Q', 'R', 'S', 'T', '-'])
#actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
#actions = np.array(['K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', '-'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30
# %%
#====================
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# %% 6. Preprocess Data and Create Labels and Features

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# %%
label_map = {label:num for num, label in enumerate(actions)}
# %%
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
# %%
# nguye@DESKTOP-42L3GL9 MINGW64 /d/Sign Language Action Regconition/Logs/train (master)
# $ tensorboard --logdir=.
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
# %%
model = Sequential()
#model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
# %%
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# %%
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])
# %%
res = model.predict(X_test)
# %%
model.save('20letters.h5')
model.load_weights('20letters.h5')
# %%
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
