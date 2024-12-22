# AI Text Detector
The AI Detector project aims to provide a robust system for identifying whether a given text is human-written or AI-generated. Unlike existing tools, our detector reduces false positives and negatives, ensuring fairness and accuracy in educational, professional, and content validation contexts.

## Table of Contents
1. [Features](#Features)
2. [Setup](#Setup)
3. [Installation](#Installation)
4. [Usage](#Usage)
5. [Contributors](#Contributors)

## Features
- High Accuracy: Improved detection rates compared to existing tools like GPT Zero.
- Fairness: Reduces the chances of misclassifying human-authored content as AI-generated.
- Customizable Thresholds: Adjustable sensitivity for different use cases.
- Real-time Analysis: Process text input and return results quickly.
- Transparency: Open-source codebase and clear explanation of model decisions.

## Setup
1. Make sure your system has Python version 3.8 or later installed.
2. Clone this directory using the code below.
     ```bash
     git clone https://github.com/helloswayamshah/AI-Detector.git
     ```
     **Note: Make sure git is installed on your computer**
3. Create a Python virtual environment to install the required dependencies. Use the command below.
     ```bash
     python -m venv <virtual_environment_name>
     ```
4. Once the virtual environment is created, follow the activation of the virtual environment for your [Windows](#Windows) or [Linux/Mac](#Linux/Mac) Machine.
     ### Windows
     ```console
     ./<virtual_environment_name>/Scripts/activate
     ```
     ### Linux/Mac
     ```bash
     source ./<virtual_environment_name>/bin/activate
     ```
5. Once the virtual environment is activated your terminal will look like shown below.
     ```
     (<virtual_environment_name>) <path_to_curr_directory>
     ```
   Now you are all set to install the AI Detector

## Installation
1. To run the AI Detector, make sure to install all the dependencies for the AI detector by running the command below:
     ### Windows
     ```console
     pip install -r .\requirements.txt
     ```
     ### Linux/Mac
     ```bash
     pip install -r ./requirements.txt
     ```
2. All the required dependencies will be installed by running the command given above, **Make sure your computer is connected to reliable internet to download all the dependencies**.
3. Make sure the spacy language model is downloaded using the command given below:
     ```bash
     python -m spacy download en_core_web_sm
     ```
You are all set to run the AI detector
## Usage
To run the AI detector run the command given below:
### Windows
```console
python .\ai_detector.py
```
### Linux/Mac
```bash
python ./ai_detector.py
```
The AI detector will start training on the training data given in the `data` directory as `TRAIN.csv`.
**Do not make any changes to the naming of the data directory, make sure to keep up with the data standards and the format in case you wish to add more training data.** 

The AI detector will train on the data using different models, and provide you with the prompt for input, Write the data you wish to classify and the detector will provide you with the result.
## Contributors
The collaborators for this project are:
1. **Swayam Shah**
     - personal-email: [helloswayamshah@gmail.com](mailto:helloswayamshah@gmail.com)
     - school-email: [sshah36@ucsc.edu](mailto:sshah36@ucsc.edu)
2. **Atharva Tawde**
3. **Jiancheng Xiong**
4. **Karthik Chaparala**
