import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import *

pre_process_Output = "../../pre_process/"
logs = "../../Logs/Sequential/"
modelType = "/Sequential/"

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

# Define the model

model = Sequential()
model.add(Dense(700, input_dim=7, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create a Tensorboard Logger
logger = TensorBoard(log_dir=logs,
                     histogram_freq=5,
                     write_graph=True)

history = model.fit(X_train, Y_train,
                    epochs=40, batch_size=192, validation_data=(X_test, Y_test),
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