import argparse
import librosa
import sys

# Parse command line arguments
parser = argparse.ArgumentParser()
# Adding input argument
parser.add_argument("-i", "--Input", help="input file path")
# Read arguments from command line
args = parser.parse_args()

if args.Input:
    test_audio = args.Input
else:
    sys.exit(1)


signal, sr = librosa.load(test_audio)
sys.stdout.write(str(sr))
