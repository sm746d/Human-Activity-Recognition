import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dropout, LSTM, Dense, K

pre_process_Output = "../../pre_process/"
logs = "../../Logs/LSTM/"
modelType = "/LSTM/"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_batches = 1392
test_batches = 352
time_sequence = 24
num_features = 7

# Load Training and testing data pre-processed in csv files
training_data_df = pd.read_csv("{}AReM_Training.csv".format(pre_process_Output))
testing_data_df = pd.read_csv("{}AReM_Testing.csv".format(pre_process_Output))

# Load X_train and X_test
X_train = training_data_df.drop('Motion', axis=1).values
X_test = testing_data_df.drop('Motion', axis=1).values

# Load Y_train and Y_test
Y_val_train = training_data_df['Motion'].values
Y_val_test = testing_data_df['Motion'].values

# Encode Y_training values and T_testing values with LabelEncoder
encoder = LabelEncoder()
Y_enc_train = encoder.fit_transform(Y_val_train)
Y_enc_test = encoder.fit_transform(Y_val_test)

# Convert Y_enc_train and Y_enc_test to categorical values
Y_train = np_utils.to_categorical(Y_enc_train)
Y_test = np_utils.to_categorical(Y_enc_test)

# Reshape train data to fit LSTM model
X_train = np.reshape(X_train, (train_batches, time_sequence, num_features))
X_test = np.reshape(X_test, (test_batches, time_sequence, num_features))

Y_train = np.reshape(Y_train, (train_batches, time_sequence, num_features))
Y_test = np.reshape(Y_test, (test_batches, time_sequence, num_features))

# Define the model
model = Sequential()
layers = [128, 128]
model.add(LSTM(layers[0], input_shape=(time_sequence, num_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(layers[1], return_sequences=True))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create a Tensorboard Logger
logger = TensorBoard(log_dir=logs,
                     histogram_freq=5,
                     write_graph=True)

history = model.fit(X_train, Y_train, epochs=40, batch_size=3, validation_data=(X_test, Y_test),
                    verbose=2, callbacks=[logger])

model_builder = tf.saved_model.builder.SavedModelBuilder("../exported_model{}".format(modelType))

inputs = {
    'input': tf.saved_model.utils.build_tensor_info(model.input)
}
outputs = {
    'motion': tf.saved_model.utils.build_tensor_info(model.output)
}

signature_def = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)

model_builder.add_meta_graph_and_variables(
    K.get_session(),
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }
)

model_builder.save()
