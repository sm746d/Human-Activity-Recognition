import pandas as pd, numpy as np, os, fnmatch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# File Loading
root = "../Data/"
pre_process_Output = "../pre_process/"
pattern = "*.csv"

cols = ['Time', 'avgrss12', 'varrss12', 'avgrss13', 'varrss13', 'avgrss23', 'varrss23']
classes = ['Bending1', 'Bending2', 'Cycling', 'Lying', 'Sitting', 'Standing', 'Walking']
features = ['Time', 'avgrss12', 'varrss12', 'avgrss13', 'varrss13', 'avgrss23', 'varrss23']
target = ['Motion']

pdBending1 = pd.DataFrame({}, columns=cols, dtype=float)
pdBending2 = pd.DataFrame({}, columns=cols, dtype=float)
pdCycling = pd.DataFrame({}, columns=cols, dtype=float)
pdLying = pd.DataFrame({}, columns=cols, dtype=float)
pdSitting = pd.DataFrame({}, columns=cols, dtype=float)
pdStanding = pd.DataFrame({}, columns=cols, dtype=float)
pdWalking = pd.DataFrame({}, columns=cols, dtype=float)

data = pd.DataFrame({}, columns=cols, dtype=float)

def trainingSet(filename, label):
    global pdBending1, pdBending2, pdCycling, pdLying, pdSitting, pdStanding, pdWalking
    temp_df = pd.read_csv(filename, skiprows=5, names=cols, skip_blank_lines=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp = pd.DataFrame(scaler.fit_transform(temp_df), columns=cols)
    if label == 'bending1':
        pdBending1 = pdBending1.append(temp)

    elif label == 'bending2':
        pdBending2 = pdBending2.append(temp)

    elif label == 'cycling':
        pdCycling = pdCycling.append(temp)

    elif label == 'lying':
        pdLying = pdLying.append(temp)

    elif label == 'sitting':
        pdSitting = pdSitting.append(temp)

    elif label == 'standing':
        pdStanding = pdStanding.append(temp)

    elif label == 'walking':
        pdWalking = pdWalking.append(temp)

label = ''
for root, dirs, files in os.walk(root):
    for filename in files:
        if fnmatch.fnmatch(filename, pattern):
            with open(os.path.join(root, filename)) as myfile:
                label = [next(myfile) for x in range(1)][0].split()[2]
            trainingSet((os.path.join(root, filename)), label)

def dataFrameCollector():
    global pdBending1, pdBending2, pdCycling, pdLying, pdSitting, pdStanding, pdWalking, data

    pdBending1['Motion'] = classes[0]
    pdBending2['Motion'] = classes[1]
    pdCycling['Motion'] = classes[2]
    pdLying['Motion'] = classes[3]
    pdSitting['Motion'] = classes[4]
    pdStanding['Motion'] = classes[5]
    pdWalking['Motion'] = classes[6]

    data = data.append(pdBending1.append(pdBending2.append(pdCycling.append(pdLying.append(
        pdSitting.append(pdStanding.append(pdWalking)))))))

dataFrameCollector()

new_index = np.arange(len(data))
data.index = new_index

# Format : X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X = data[features]
y = data[target]

X_training, X_testing, Y_training, Y_testing = train_test_split(X, y, test_size=0.20)

# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.

training_data_df = pd.concat([X_training, Y_training], axis=1)
test_data_df = pd.concat([X_testing, Y_testing], axis=1)

training_data_df.to_csv("{}AReM_Training.csv".format(pre_process_Output), index=False)
test_data_df.to_csv("{}AReM_Testing.csv".format(pre_process_Output), index=False)