import os
import re
import pandas as pd
import nltk
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')

pwd = os.getcwd()
path = pwd +'/data/robot-ai-all-public-v2.csv'


# Assuming df is your DataFrame and 'Paragraph' is your column with text
df = pd.read_csv(path)  # replace 'your_file.csv' with your file path

spell = SpellChecker()

def correct_spelling_and_remove_symbols(text):
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Remove special symbols
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    
    # Correct spelling
    words = text.split()
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word is None:
            corrected_word = word  # use the original word if no correction is found
        corrected_words.append(corrected_word)
    return " ".join(corrected_words)

df['Paragraph'] = df['Paragraph'].apply(correct_spelling_and_remove_symbols)


# Save the corrected DataFrame to a new CSV file
df.to_csv(pwd+'/data/robot-ai-all-public-v3.csv', index=False)


# Assuming df is your DataFrame and 'Paragraph' is your column of text
df = pd.read_csv(pwd + '/data/robot-ai-all-public-v3.csv')


# Tokenize the text into individual words
df['Words'] = df['Paragraph'].apply(word_tokenize)

# Flatten the list of words and remove stop words
words = [word for sublist in df['Words'].tolist() for word in sublist if word not in stopwords.words('english')]

# Count the frequency of each word
word_freq = Counter(words)

# Convert the Counter object to a DataFrame
word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
word_freq_df.columns = ['Word', 'Frequency']

# Save the DataFrame to a csv file
word_freq_df.to_csv(pwd + '/data/robot-ai-all-public-v4.csv', index=False)
