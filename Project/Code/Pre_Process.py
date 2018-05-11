import pandas as pd, numpy as np, os, fnmatch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import *

# File Loading
root = "../Data/"
pre_process_Output = "../pre_process/"
pattern = "*.csv"

# Encode Labels, defining parameters
le = LabelEncoder()
cols = ['Time', 'avgrss12', 'varrss12', 'avgrss13', 'varrss13', 'avgrss23', 'varrss23']
all_cols = ['Time', 'avgrss12', 'varrss12', 'avgrss13', 'varrss13', 'avgrss23', 'varrss23', 'Motion']
classes = ['Bending1', 'Bending2', 'Cycling', 'Lying', 'Sitting', 'Standing', 'Walking']
classes = le.fit_transform(classes)
features = ['Time', 'avgrss12', 'varrss12', 'avgrss13', 'varrss13', 'avgrss23', 'varrss23']
target = ['Motion']

# Initializing DataFrames
pdBending1 = pd.DataFrame({}, columns=cols, dtype=float)
pdBending2 = pd.DataFrame({}, columns=cols, dtype=float)
pdCycling = pd.DataFrame({}, columns=cols, dtype=float)
pdLying = pd.DataFrame({}, columns=cols, dtype=float)
pdSitting = pd.DataFrame({}, columns=cols, dtype=float)
pdStanding = pd.DataFrame({}, columns=cols, dtype=float)
pdWalking = pd.DataFrame({}, columns=cols, dtype=float)

train = pd.DataFrame({}, columns=all_cols, dtype=float)
test = pd.DataFrame({}, columns=all_cols, dtype=float)

data = pd.DataFrame({}, columns=cols, dtype=float)


def trainingSet(filename, label):
    global pdBending1, pdBending2, pdCycling, pdLying, pdSitting, pdStanding, pdWalking, train, test
    temp_df = pd.read_csv(filename, skiprows=5, names=cols, skip_blank_lines=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp_df = pd.DataFrame(scaler.fit_transform(temp_df), columns=cols)

    if label == 'bending1':
        temp_df['Motion'] = classes[0]

    elif label == 'bending2':
        temp_df['Motion'] = classes[1]

    elif label == 'cycling':
        temp_df['Motion'] = classes[2]

    elif label == 'lying':
        temp_df['Motion'] = classes[3]

    elif label == 'sitting':
        temp_df['Motion'] = classes[4]

    elif label == 'standing':
        temp_df['Motion'] = classes[5]

    elif label == 'walking':
        temp_df['Motion'] = classes[6]

    train_temp, test_temp = train_test_split(temp_df, train_size=0.8)
    if (train_temp.isnull().values.any()):
        print(filename)
    # if train_temp.Motion.isnull().empty:
    #    print(filename)
    train_temp = train_temp.sort(["Time"])
    test_temp = test_temp.sort(["Time"])

    train = train.append(train_temp)
    test = test.append(test_temp)

label = ''
for root, dirs, files in os.walk(root):
    for filename in files:
        if fnmatch.fnmatch(filename, pattern):
            with open(os.path.join(root, filename)) as myfile:
                label = [next(myfile) for x in range(1)][0].split()[2]
            trainingSet((os.path.join(root, filename)), label)

# Remove erroneous values from Training set
train = train.dropna()

# Re-index values
new_index = np.arange(len(train))
train.index = new_index

# Add dummy row to Training set to fit better in LSTM model
data = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
d = pd.DataFrame(data, columns = train.columns)
train = train.append(d)
new_index = np.arange(len(train))
train.index = new_index

# Training data shuffling
training_np = np.array(train).reshape(348, 96, 8)
idx = np.random.randint(348, size=348)
arr2 = training_np[idx, :]
arr2 = arr2.reshape(33408, 8)
training = pd.DataFrame(arr2, columns=train.columns)

# Testing data Shuffling
testing_np = np.array(test).reshape(88, 96, 8)
idx = np.random.randint(88, size=88)
arr2 = testing_np[idx, :]
arr2 = arr2.reshape(8448, 8)
testing = pd.DataFrame(arr2, columns=test.columns)

# Calculating inverse transform of Motion
training['Motion'] = le.inverse_transform(training['Motion'].astype(int))
testing['Motion'] = le.inverse_transform(testing['Motion'].astype(int))

# Save pre-processed files to pre_process folder
training.to_csv("../pre_process/AReM_Training.csv", index=False)
testing.to_csv("../pre_process/AReM_Testing.csv", index=False)
