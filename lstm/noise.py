import numpy as np
import pandas as pd
import random
import pandas as pd
import nltk
from nltk.corpus import wordnet

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("/bbc-text.csv")

# Define a function to add random noise to a string
def add_noise(text, noise_level):
    # Split the text into words
    words = text.split()
    # Loop over the words and add noise to each one
    for i in range(len(words)):
        # Calculate the probability of adding noise to this word
        p = random.random()
        if p < noise_level:
            # Choose a random type of noise to add
            noise_type = random.choice(["swap", "insert", "delete"])
            # If "swap" noise is chosen, swap two random characters in the word
            if noise_type == "swap" and len(words[i]) > 1:
                idx1, idx2 = np.random.choice(len(words[i]), 2, replace=False)
                words[i] = words[i][:idx1] + words[i][idx2] + words[i][idx1+1:idx2] + words[i][idx1] + words[i][idx2+1:]
            # If "insert" noise is chosen, insert a random character into the word
            elif noise_type == "insert":
                idx = np.random.randint(len(words[i])+1)
                char = random.choice(["!", "?", ".", ",", "", ")", "(", "#", "@"])
                words[i] = words[i][:idx] + char + words[i][idx:]
            # If "delete" noise is chosen, delete a random character from the word
            elif noise_type == "delete" and len(words[i]) > 1:
                idx = np.random.randint(len(words[i]))
                words[i] = words[i][:idx] + words[i][idx+1:]
    # Join the words back together into a string
    noisy_text = " ".join(words)
    return noisy_text

# Define a function to add spelling errors
def add_spelling_errors(text, error_rate):
    words = nltk.word_tokenize(text)
    noisy_words = []
    for word in words:
        if len(word) > 3 and nltk.probability.FreqDist(word.lower()).N() > 2:
            if error_rate > 0 and nltk.probability.FreqDist(word.lower()).N() > 2:
                noisy_word = ""
                for i in range(len(word)):
                    if (nltk.probability.FreqDist(word[i]).N() > 2) and (nltk.probability.FreqDist(word[i+1]).N() > 2):
                        if np.random.rand() < error_rate:
                            noisy_word += word[i+1] + word[i]
                        else:
                            noisy_word += word[i]
                    else:
                        noisy_word += word[i]
                noisy_words.append(noisy_word)
            else:
                noisy_words.append(word)
        else:
            noisy_words.append(word)
    return ' '.join(noisy_words)

# Define a function to replace words
def replace_words(text, replacement_prob):
    words = nltk.word_tokenize(text)
    for i in range(len(words)):
        if np.random.rand() < replacement_prob:
            word_synonyms = []
            for syn in wordnet.synsets(words[i]):
                for lemma in syn.lemmas():
                    word_synonyms.append(lemma.name())
            if word_synonyms:
                new_word = np.random.choice(word_synonyms)
                words[i] = new_word
    return ' '.join(words)


# Add spelling errors and replace words to the text column in your dataset
df['text'] = df['text'].apply(lambda x: add_spelling_errors(x, error_rate=2.5))
df['text'] = df['text'].apply(lambda x: replace_words(x, replacement_prob=2.5))
df['text'] = df['text'].apply(lambda x: add_noise(x, 3))

# Save the modified dataset
df.to_csv('./modified_bbc_articles1.csv', index=False)
