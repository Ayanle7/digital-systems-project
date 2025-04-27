with open('fdrpearlharbor.txt', 'r') as file:
    transcript = file.read()

words = transcript.split()

starting_index = 1453
ending_index =  1488

audio_files_count = ending_index - starting_index + 1

words_per_segment = len(words) // audio_files_count
remaining_words = len(words) % audio_files_count  

segments = []
current_index = 0
for i in range(audio_files_count):

    extra_word = 1 if i < remaining_words else 0
    segment_length = words_per_segment + extra_word
    segments.append(words[current_index:current_index + segment_length])
    current_index += segment_length

with open('fdr.txt', 'a') as output_file:  
    for i, segment in enumerate(segments):
        audio_file = f"output_{starting_index + i:03d}.wav"  
        segment_text = ' '.join(segment)
        output_file.write(f"{audio_file}: \"{segment_text}\"\n")

print("Transcript appended and saved as 'fdr.txt'")

