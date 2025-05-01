import pytest
import torch
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from voiceApp import generate_waveform_and_spectrogram  


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("Ayanle7/fdr_model4")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speaker_embedding = torch.tensor(np.load("speaker_embedding.npy")).unsqueeze(0)


def test_valid_output_shapes():
    text = "Hello, world!"
    waveform, spectrogram = generate_waveform_and_spectrogram(text, processor, model, vocoder, speaker_embedding)

    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 1
    assert waveform.shape[0] > 0

    assert isinstance(spectrogram, np.ndarray)
    assert spectrogram.ndim == 2
    assert spectrogram.shape[0] > 0


def test_empty_text_raises_error():
    with pytest.raises(ValueError, match="Input text cannot be empty."):
        generate_waveform_and_spectrogram("   ", processor, model, vocoder, speaker_embedding)

def test_waveform_range():
    text = "Testing waveform range."
    waveform, _ = generate_waveform_and_spectrogram(text, processor, model, vocoder, speaker_embedding)

    assert waveform.dtype == np.float32
    assert np.all(np.isfinite(waveform)), "Waveform contains non-finite values."


def test_audio_length():
    text = "This is a test sentence."
    waveform, _ = generate_waveform_and_spectrogram(text, processor, model, vocoder, speaker_embedding)

    assert waveform.shape[0] > 0, "Waveform has no length."


def test_empty_text():
    text = ""
    speaker_embedding = torch.rand(1, 512)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("Ayanle7/fdr_model4")
    inputs = processor(text=text, return_tensors="pt")
    try:
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding)
        print("test_empty_text: Passed (handled gracefully)")
    except Exception as e:
        print(f"test_empty_text: Failed with error: {e}")

def test_special_characters():
    text = "Wait... what?! Really?! — No way!!!"
    speaker_embedding = torch.rand(1, 512)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("Ayanle7/fdr_model4")
    inputs = processor(text=[text], return_tensors="pt")  
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding)
    assert speech.ndim in [1, 2]

def test_numeric_input():
    text = "The price is 19.99 dollars and delivery is in 2 days."
    speaker_embedding = torch.rand(1, 512)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("Ayanle7/fdr_model4")
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding)
    assert speech.ndim in [1, 2]
    print("test_numeric_input passed.")

def test_unicode_input():
    text = "Hello 😊 How are you?"
    speaker_embedding = torch.rand(1, 512)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("Ayanle7/fdr_model4")
    inputs = processor(text=text, return_tensors="pt")
    try:
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding)
        print("test_unicode_input passed.")
    except Exception as e:
        print(f"test_unicode_input failed: {e}")


def test_invalid_speaker_embedding_shape():
    bad_embedding = torch.randn(1, 5)  

    with pytest.raises(Exception):  
        generate_waveform_and_spectrogram("Valid text", processor, model, vocoder, bad_embedding)

def test_invalid_model_or_processor():

    invalid_model_path = "nonexistent/model"

    with pytest.raises(OSError):
        SpeechT5ForTextToSpeech.from_pretrained(invalid_model_path)

    invalid_processor_path = "nonexistent/processor"

    with pytest.raises(OSError):
        SpeechT5Processor.from_pretrained(invalid_processor_path)