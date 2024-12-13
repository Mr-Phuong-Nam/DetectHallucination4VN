import wikipedia
import pandas as pd
from ast import literal_eval
import logging as log
import os
# Set language to Vietnamese
wikipedia.set_lang("vi")

# Function to fetch summary of a biography
def get_summary(name):
    try:
        summary = wikipedia.summary(name, sentences=3)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation Error: {e}"
    except wikipedia.exceptions.PageError:
        return f"Page '{name}' not found."
    except Exception as e:
        return f"Error: {e}"

file_names = ["geographical_places_dataset_2.csv", "historical_figures_dataset.csv"]

working_file = input("Enter the file name: (1 - Geographical Places, 2 - Historical Figures): ")

if working_file == "1":
    file_name = file_names[0]
elif working_file == "2":
    file_name = file_names[1]
else:
    print("Invalid input. Please try again.")
    exit()

# Read the dataset
file_name = os.path.join("./File", file_name)
df = pd.read_csv(file_name)
working_state = input("Enter the working state (1 - Annotate, 2 - Review): ")


# title,gemini_text,gemini_sentences,annotation,gemini_text_samples
# covert gemini_sentences and annotation to list
df['gemini_sentences'] = df['gemini_sentences'].apply(literal_eval)
df['annotation'] = df['annotation'].apply(literal_eval)
if 'annotation_note' not in df.columns:
    df['annotation_note'] = df['annotation'].apply(lambda x: [None for _ in x])
else:
    df['annotation_note'] = df['annotation_note'].apply(literal_eval)
df['gemini_text'] = df['gemini_text'].apply(lambda x: x.strip())


if working_state == "1":
    print("Annotating...")
    if len(df[df['annotation'].apply(lambda x: any([y is None for y in x]))]) == 0:
        print("All rows have been annotated.")
        exit()
    try:
        for index, row in df.iterrows():
            if all([x is not None for x in row['annotation']]):
                continue
            print("===============================================")
            print(f"[+] Title: {row['title']}")
            print(f"[+] Gemini Text: \n{row['gemini_text']}")
            # print(f"[+] Summary: \n{get_summary(row['title'])}")
            for i, sentences in enumerate(row['gemini_sentences']):
                if row['annotation'][i] is None:
                    print(f"\t[-] Sentence {i+1}: {sentences}")
                    annotation = input("\t[-->] Annotation: ")
                    if annotation != "0":
                        note = input("\t[-->] Note: ")
                        df.at[index, 'annotation_note'][i] = note
                    df.at[index, 'annotation'][i] = annotation
    except KeyboardInterrupt:
        print("Interrupted by user.")
        print("Saving the dataset...")
        df.to_csv(file_name, index=False)
        print("Dataset saved.")
        exit()
elif working_state == "2":
    print("Reviewing...")
    try:
        print("List of titles:")
        for title in df['title']:
            print(f"\t[-] {title}")
        tittle = input("Enter the title to review: ")
        for index, row in df.iterrows():
            if row['title'] != tittle:
                continue
            print("===============================================")
            print(f"[+] Title: {row['title']}")
            print(f"[+] Gemini Text: \n{row['gemini_text']}")
            while True:
                num = int(input("Enter the sentence number to review (0 to skip)"))
                if num == 0:
                    break
                if len(row['annotation']) < num:
                    print("Invalid sentence number.")
                    continue
                print(f"\t[-] Sentence {num}: {row['gemini_sentences'][num-1]}")
                print(f"\t[-] Annotation: {row['annotation'][num-1]}")
                print(f"\t[-] Note: {row['annotation_note'][num-1]}")
                annotate = input("\t[-->] Annotate? (Y/N): ")
                if annotate.lower() == "n":
                    continue
                annotation = input("\t[-->] New Annotation: ")
                if annotation != "0":
                    note = input("\t[-->] Note: ")
                    df.at[index, 'annotation_note'][num-1] = note
                df.at[index, 'annotation'][num-1] = annotation
    except KeyboardInterrupt:
        print("Interrupted by user.")
        print("Saving the dataset...")
        df.to_csv(file_name, index=False)
        print("Dataset saved.")
        exit()
else:
    print("Invalid input. Please try again.")
    exit()



print("All rows have been annotated.")
print("Saving the dataset...")
df.to_csv(file_name, index=False)
print("Dataset saved.")