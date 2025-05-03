# Fine-Tuned Speech Synthesis with Streamlit

This project leverages the **SpeechT5** model for text-to-speech generation using fine-tuned speaker embeddings. It allows users to input text, which is then converted into speech, displayed with a spectrogram, and provided for download.

## Requirements

Before running this project, make sure you have Python 3.12 installed, and you can set up your environment by running:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # For macOS/Linux
# .venv\Scripts\activate   # For Windows
pip install -r requirements.txt
```

# Running the App

1. Activate your virtual environment:

source .venv/bin/activate  # For macOS/Linux

2. To start the app, run:

streamlit run streamlit_tts_app.py

# Requirements file

The project requires the following Python libraries:

* streamlit
* torch
* transformers
* numpy
* matplotlib
* soundfile
* pytest

To run the unit test, use the following command:

```bash
pytest unitTest.py
