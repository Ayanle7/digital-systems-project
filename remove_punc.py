import re

input_path = "./fdr.txt"
output_path = "./fdr_cleaned.txt"
with open(input_path, 'r') as file:
    lines = file.readlines()

cleaned_lines = []
for line in lines:
    if ':' in line:
        identifier, sentence = line.split(':', 1)
        sentence = sentence.strip()

        # Remove outer quotes if present
        sentence = sentence.strip('"')

        # If sentence ends with a period, preserve it
        if sentence.endswith('.'):
            main_part = sentence[:-1]
            main_part = re.sub(r'[,:;]|--|—|-', '', main_part)
            cleaned_sentence = main_part.strip() + '.'
        else:
            cleaned_sentence = re.sub(r'[,:;]|--|—|-', '', sentence).strip()

        cleaned_lines.append(f'{identifier.strip()}: "{cleaned_sentence}"\n')
    else:
        cleaned_lines.append(line)

# Write the result to a new file
with open(output_path, 'w') as out_file:
    out_file.writelines(cleaned_lines)

print(f"Cleaned output saved to {output_path}")