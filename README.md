# Automatic Speech Detection Model

## Metrics for Test Data

1. **Accuracy - 96.35%**
2. **F1 Score - 94.60%**
3. Loss = 0.3370

## main.py

Loads the ASR model, predicts the frames with speech, and calculates the start & end times of time segments with speech

### Usage:

```
python main.py -i test.wav -o prediction.csv
```

### Parameters:

Set `noise_threshold` and `pause_threshold` to appropriate values depending on the requied amount of precision.

## Assignment 4.ipynb

The code used to genrate and save the ASR model

1.  We first use the given start and end times to generate a class vector with appropriate class value for each smaller window.
2.  We then caculate the MFCCs, delta MFCCs and delta^2 MFCCs of the audio in those windows and concatenate them to get our feature matrix.
3.  We then split it into train and test data and train the model.
4.  Lastly we use the model to predict speech segment in each window and then convert the predicted vector into the required form with start and end times.

## Note:

The .ipynb file uses the `tensorflow_addons` package to calculate F1 Score. This doesn't run on Google Colab and hence has been commented out after calculation. Please add uncomment its instance and add it to the metrics in `model.compile` to find F1 Score
