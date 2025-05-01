import streamlit as st
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import set_seed, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf

seed_value = 555
set_seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Load the models and processor 
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("Ayanle7/fdr_model4")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embedding
speaker_embedding_path = './speaker_embedding.npy'
speaker_embeddings = np.load(speaker_embedding_path)
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)


def generate_waveform_and_spectrogram(text, processor, model, vocoder, speaker_embeddings):
    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    inputs = processor(text=text, return_tensors="pt")
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

    with torch.no_grad():
        waveform = vocoder(spectrogram)

    waveform_np = waveform.squeeze().cpu().numpy()
    if waveform_np.ndim > 1:
        waveform_np = waveform_np[0]
    waveform_np = waveform_np.astype(np.float32)

    spectrogram_np = spectrogram.squeeze().cpu().numpy().T

    return waveform_np, spectrogram_np


def main():
    st.title("Fine-Tuned Speech Synthesis")

    editable_text = st.text_area("Enter text for speech generation:", height=150)

    if "waveform_np" not in st.session_state:
        st.session_state.waveform_np = None
        st.session_state.spectrogram = None

    if st.button("Generate Speech"):
        with st.spinner("Generating speech..."):
            try:
                waveform_np, spectrogram_np = generate_waveform_and_spectrogram(
                    editable_text, processor, model, vocoder, speaker_embeddings
                )

                st.session_state.waveform_np = waveform_np
                st.session_state.spectrogram = spectrogram_np
                st.success("Speech generated! You can adjust playback speed by right-clicking the audio and selecting 'Speed'.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.waveform_np is not None:
        # Spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(st.session_state.spectrogram, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title("Generated Spectrogram")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Waveform
        fig_wave, ax_wave = plt.subplots(figsize=(10, 2))
        ax_wave.plot(st.session_state.waveform_np, color='b')
        ax_wave.set_title("Waveform (Amplitude over Time)")
        ax_wave.set_xlabel("Time (samples)")
        ax_wave.set_ylabel("Amplitude")
        st.pyplot(fig_wave)

        # Audio playback and download
        sample_rate = 16000
        st.audio(st.session_state.waveform_np, format="audio/wav", sample_rate=sample_rate)

        output_path = "output.wav"
        sf.write(output_path, st.session_state.waveform_np, sample_rate)
        with open(output_path, "rb") as f:
            st.download_button("Download Audio", f, file_name="generated_speech.wav")

if __name__ == "__main__":
    main()