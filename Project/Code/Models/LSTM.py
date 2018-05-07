import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import *

pre_process_Output = "../../pre_process/"
logs = "../../Logs/LSTM/"
modelType = "/LSTM/"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load Training and testing data pre-processed in csv files
training_data_df = pd.read_csv("{}AReM_Training.csv".format(pre_process_Output))
testing_data_df = pd.read_csv("{}AReM_Testing.csv".format(pre_process_Output))

# Load X_train and X_test
X_train = training_data_df.drop('Motion', axis=1).values
X_test = testing_data_df.drop('Motion', axis=1).values

# Load Y_train and Y_test
Y_val_train = training_data_df[['Motion']].values
Y_val_test = testing_data_df[['Motion']].values

# Encode Y_training values and T_testing values with LabelEncoder
encoder = LabelEncoder()
Y_enc_train = encoder.fit_transform(Y_val_train)
Y_enc_test = encoder.fit_transform(Y_val_test)

# Convert Y_enc_train and Y_enc_test to categorical values
Y_train = np_utils.to_categorical(Y_enc_train)
Y_test = np_utils.to_categorical(Y_enc_test)

# Reshape train data to fit LSTM model
X_train = X_train.reshape(11, 3037, 7)
Y_train = Y_train.reshape(11, 3037, 7)

# Drop rows with row number more than 3037 for Validation set to fit LSTM cell and reshape arrays
X_test = np.delete(X_test, np.s_[3037:], axis=0).reshape(1, 3037, 7)
Y_test = np.delete(Y_test, np.s_[3037:], axis=0).reshape(1, 3037, 7)

# Define the model
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(3037, 7), return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(7, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

# Create a Tensorboard Logger
logger = TensorBoard(log_dir=logs,
                     histogram_freq=5,
                     write_graph=True)

history = model.fit(X_train, Y_train, epochs=10, batch_size=100, validation_data=(X_test, Y_test),
                    verbose=2, callbacks=[logger])


model_builder = tf.saved_model.builder.SavedModelBuilder("../exported_model".format(modelType))

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