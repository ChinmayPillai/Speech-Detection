import argparse
import sys
import csv
import librosa
import numpy as np
import tensorflow as tf
# from tensorflow_addons.metrics import F1Score

# Test on new audio flle
test_audio = 'mic_M01_sa1.wav'
# Define the path to the CSV file
csv_file = 'prediction_test.csv'
# Define threshhold below which speech segment is considered as noise and ignored
noise_threshold = 0.05
# Define threshold of pause below which speech is considered as one single segment
pause_threshold = 0.25


# Parse command line arguments
parser = argparse.ArgumentParser()
# Adding input argument
parser.add_argument("-i", "--Input", help="input file path")
# Adding output argument
parser.add_argument("-o", "--Output", help="output file path")
# Read arguments from command line
args = parser.parse_args()

if args.Input:
    test_audio = args.Input
else:
    sys.exit(1)

if args.Output:
    csv_file = args.Output


# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(78, activation='relu', input_shape=(39,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(156, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Restore the weights
model.load_weights('./checkpoints/weights')

# Load audio
signal_test, sr_test = librosa.load(test_audio)

# Calculate MFCCs, delta MFCCs and delta delta MFCCs and concatenate them to get feature vector
mfcc_test = librosa.feature.mfcc(y=signal_test, n_mfcc=13, sr=sr_test)
del_mfcc_test = librosa.feature.delta(mfcc_test)
del2_mfcc_test = librosa.feature.delta(mfcc_test, order=2)
mfcc_features_test = np.concatenate((mfcc_test, del_mfcc_test, del2_mfcc_test))
mfcc_features_test = np.transpose(mfcc_features_test)

# Predict output from features
prediction_test = model.predict(mfcc_features_test)

# Supress and amplify predictions to 0 and 1
prediction_test[prediction_test >= 0.5] = 1
prediction_test[prediction_test < 0.5] = 0
prediction_test.astype(int)


# Generate segments of speech from the predictions for each small segment
previous_frame = 0
start_time = 0
stepLength = 0.02321981922

data = [["", "Start", "End"]]
idx = 0

for i in range(len(prediction_test)):

    current_time = i*stepLength
    current_frame = prediction_test[i]
    # If previous frame has speech
    if previous_frame == 1:

        # and current frame doesn't have speech,
        # and falls outside noise threshold
        if current_frame == 0 and current_time-start_time >= noise_threshold:
            # print("Start Time: ", start_time, "End Time: ", current_time)
            data.append([idx, start_time, current_time])
            idx += 1

    # If previous frame doesn't have speech
    # and current frame does have speech
    # and falls outside pause threshold
    elif current_frame == 1:
        start_time = current_time
        if idx != 0 and start_time - float(data[-1][2]) < pause_threshold:
            start_time = float(data[-1][1])
            data.pop()
            idx -= 1

    previous_frame = current_frame

# Get last speech segment if last frame also has speech
if previous_frame == 1:
    if len(prediction_test) * stepLength - start_time >= noise_threshold:
        data.append([idx, start_time, current_time])


# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the data to the CSV file row by row
    for row in data:
        writer.writerow(row)
