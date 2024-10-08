# ASR Correction Agent

## Overview

This project implements an agent that corrects errors in text generated by Automatic Speech Recognition (ASR) systems. The agent uses a **local search algorithm** to refine ASR outputs by performing **phoneme corrections** and **word insertions** based on a **phoneme table** and a **vocabulary list**. The objective is to improve the accuracy of the transcription using a cost function, which evaluates how well the text matches the original audio.

## How It Works

The ASR Correction Agent takes input text in uppercase letters that might contain errors caused by misrecognized phonemes or missing words. The agent uses the following approach to correct the errors:

1. **Phoneme Corrections**: The agent looks for potential errors based on a provided phoneme table. The table contains a list of incorrect phonemes and their possible corrections. The agent generates neighbors by replacing incorrect phonemes within each word in the sentence.

2. **Word Insertions**: After phoneme corrections, the agent attempts to improve the sentence by inserting words from the vocabulary at the beginning or end of the sentence, where words might have been missed during the initial ASR transcription.

3. **Cost Function**: A cost function evaluates how close the corrected sentence is to the original audio. The agent explores different corrections and selects the one with the lowest cost.

4. **Local Search Algorithm**: The agent runs for a set number of iterations, refining the sentence by exploring **neighboring states** (sentences generated through phoneme corrections and word insertions). The state with the lowest cost is selected as the best correction.

## Project Structure
```
ASR_Correction_Agent
├── data/
│   ├── phoneme_table.json  # Phoneme substitution table with possible phoneme errors.
│   ├── vocabulary.json     # List of words that could be inserted into the sentence.
│   ├── data.pkl            # Sample data containing audio, incorrect text, and corrected text.
├── driver.py               # Script for running the agent on the provided data.
├── environment.yml         # Conda environment setup for the project.
├── shorten.py              # Script for reducing input audio size for testing purpose.
└── solution.py             # Main implementation of the ASR correction agent.
```


## Key Features

1. **Local Search Algorithm**: Efficiently refines ASR output by exploring phoneme corrections and word insertions.
2. **Phoneme Correction**: Identifies and corrects phoneme errors based on a provided phoneme table.
3. **Word Insertion**: Adds missing words at the start or end of a sentence based on a vocabulary list.
4. **Cost Optimization**: Uses a pre-defined cost function to guide the search towards accurate corrections.
