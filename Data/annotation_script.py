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

file_names = ["geographical_places_dataset_2.csv", "historical_figures_dataset.csv", "Vietnamese_hallucination_34_annotated.csv"]

working_file = input("Enter the file name: (1 - Geographical Places, 2 - Historical Figures, 3 - Vietnamese Hallucination): ")

if working_file == "1":
    file_name = file_names[0]
elif working_file == "2":
    file_name = file_names[1]
elif working_file == "3":
    file_name = file_names[2]
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
    double_check_df = pd.DataFrame(columns=['title', 'sentence', 'old_annotation', 'new_annotation', 'note'])
    try:
        
        while True:
            print("List of titles:")
            for i, title in enumerate(df['title']):
                print(f"\t[{i}] {title}")
            index = int(input("Enter the index of the title to review: "))
            print("===============================================")
            print(f"[+] Title: {df.at[index, 'title']}")
            print(f"[+] Gemini Text: \n{df.at[index, 'gemini_text']}")
            for i, sentences in enumerate(df.at[index, 'gemini_sentences']):
                print(f"\t[-] Sentence {i+1}: {sentences}")
                print(f"\t[-->] Annotation: {df.at[index, 'annotation'][i]}")
                print(f"\t[-->] Note: {df.at[index, 'annotation_note'][i]}")
                is_continue = input("Continue? (y/n): ")
                if is_continue == "n":
                    continue    
                elif is_continue == "y":
                    new_annotation = input("New annotation: ")
                    note = input("Note: ")
                    double_check_df = pd.concat([double_check_df, pd.DataFrame([{'title': df.at[index, 'title'], 'sentence': sentences, 'old_annotation': df.at[index, 'annotation'][i], 'new_annotation': new_annotation, 'note': note}])], ignore_index=True)
                    
        
    except KeyboardInterrupt:
        print("Interrupted by user.")
        print("Saving the dataset...")
        double_check_df.to_csv("double_check.csv", index=False)
        print("Dataset saved.")
        exit()
else:
    print("Invalid input. Please try again.")
    exit()



print("All rows have been annotated.")
print("Saving the dataset...")
df.to_csv(file_name, index=False)
print("Dataset saved.")