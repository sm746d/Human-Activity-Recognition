from oauth2client.client import GoogleCredentials
import googleapiclient.discovery

# Change this values to match your project
PROJECT_ID = "activityrec1"
MODEL_NAME = "recognise"
CREDENTIALS_FILE = "credentials.json"

# These are the values we want a prediction for
inputs_for_prediction = [
    {"input": [0.05636743215031316,0.971231866240472,0.5,0.4150943396226415,0.21588089330024812,0.6153846153846154,0.2225705329153605]}
]

# Connect to the Google Cloud-ML Service
credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
service = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)

# Connect to our Prediction Model
name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
response = service.projects().predict(
    name=name,
    body={'instances': inputs_for_prediction}
).execute()

# Report any errors
if 'error' in response:
    raise RuntimeError(response['error'])

# Grab the results from the response object
results = response['predictions']

# Print the results!
print(results)
