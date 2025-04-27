import os
import json
import re
import wave

audio_dir = "./wavs_processed"
fdr_file = "./fdr_cleaned.txt"
output_jsonl = "./metadata_cleaned.jsonl"

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def get_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return round(frames / float(rate), 2)
    except Exception as e:
        print(f"Warning: Could not get duration for {file_path} - {str(e)}")
        return 0.0

with open(fdr_file, "r", encoding="utf-8") as f:
    fdr_lines = f.readlines()


results = []
missing_transcriptions = []
error_files = []

audio_files = sorted(
    [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")],
    key=lambda x: int(re.search(r"(\d+)", x).group(1))
)

for file in audio_files:

    file_path = os.path.join(audio_dir, file)
    match = re.search(r"(\d+)", file)
    
    if not match:
        error_files.append(f"{file}: No numeric ID found")
        continue

    file_id = match.group(1)
    matching_line = next((line for line in fdr_lines if file_id in line), None)

    if not matching_line:
        missing_transcriptions.append(file)
        continue

    quote_match = re.search(r'"(.*?)"', matching_line)
    raw_text = quote_match.group(1) if quote_match else matching_line.strip()

    try:
        duration = get_duration(file_path)
        normalized_text = clean_text(raw_text)

        results.append({
            "file_name": file,
            "duration": duration,
             "raw_text": raw_text,
            "normalized_text": normalized_text,
        })

    except Exception as e:
        error_files.append(f"{file}: {str(e)}")


with open(output_jsonl, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


if missing_transcriptions or error_files:
    with open("processing_errors.log", "w") as f:
        f.write("Missing transcriptions:\n" + "\n".join(missing_transcriptions) + "\n\n")
        f.write("Error files:\n" + "\n".join(error_files))

print(f"Processed {len(results)} files successfully")
print(f"Missing transcriptions: {len(missing_transcriptions)}")
print(f"Files with errors: {len(error_files)}")
print(f"Output saved to {output_jsonl}")