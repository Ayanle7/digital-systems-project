import os
import librosa
import soundfile as sf
import numpy as np


def remove_all_silence(audio, sr, top_db=20, frame_length=2048, hop_length=512):
 
    intervals = librosa.effects.split(
        audio, 
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    

    trimmed_audio = np.concatenate([
        audio[start:end] for start, end in intervals
    ])
    
    return trimmed_audio

input_dir = "./wavs"
output_dir = "./wavs_processed"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        audio, sr = librosa.load(input_path, sr=None)
        

        processed_audio = remove_all_silence(
            audio, 
            sr,
            top_db=25, 
            frame_length=1024, 
            hop_length=256
        )
        
    
        orig_dur = len(audio)/sr
        new_dur = len(processed_audio)/sr
        print(f"{filename}: {orig_dur:.2f}s → {new_dur:.2f}s")
        
        sf.write(output_path, processed_audio, sr)