# 2023_AI-FinalProject
NYCU Spring 2023 - Introduction to Artificial Intelligence - Generating Simple Melody


# Melody Generator

This project is an AI-based melody generator that uses deep learning techniques to create new melodies based on a dataset of MIDI files. The model is trained on a corpus of melodies and learns the patterns and structures within the music to generate new, unique melodies.

## Overview

The main file of this project is Melody_Generator.py, which contains the following components:

1. Data preprocessing: The script imports a dataset of MIDI files, processes them to extract the melody notes, and creates a corpus of notes. It also filters out rare notes and creates a mapping between the notes and their corresponding indices.

2. Model definition: The project uses a deep learning model based on Long Short-Term Memory (LSTM) networks. The model consists of two LSTM layers, dropout layers, and fully connected layers.

3. Training: The model is trained on the preprocessed data using the Adamax optimizer and Cross-Entropy Loss as the loss function. The training process is carried out for 100 epochs, and the loss is printed for each epoch.

4. Music generation: After training, the model generates new melodies by predicting the next note in a sequence. The generated melody is then converted into a MIDI file and can be visualized using the music21 library.

## Dependencies

This project requires the following Python libraries:

- numpy
- torch
- music21
- sklearn
- matplotlib
- seaborn

## Usage

To run the Melody_Generator.py script, simply execute the following command:
python Melody_Generator.py

After the training process is complete, the script will generate a new melody and save it as a MIDI file named Melody_Generated.mid. You can also visualize the generated melody using the show() function provided in the script.

## Example Output

The generated melody will be a list of notes, such as:
['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', ...]

This list of notes can be converted into a MIDI file or visualized using the music21 library.
