from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import spacy
from wordcloud import WordCloud

######## Data Preprocessing ########

def get_spacy_text(text_data,nlp):
    """
    Preprocesses the text data into spaCy doc and returns a list of spaCy docs.

    Args:
        text_data (list): A list of strings.
        nlp (spacy.lang): A spaCy language model.

    Returns:
        list: A list of spaCy docs.
    """
    spacy_text = []
    for i in text_data:
        #Strings are normalized to be all lowercase
        spacy_text.append(nlp(i.lower()))
    return spacy_text

def sentencize(spacy_files: list)->list:
    """
    Segments the text data into sentences and preprocesses the sentences by removing stop words, punctuation, and spaces.
    """
    segmented_files = [] 
    preprocessed_segmented_files = []
    #Segmenting the text data into sentences
    for i in spacy_files:
        segmented_files.append([sent for sent in i.sents]) 
    # Removing stop words, punctuation, and spaces from the sentences
    for i in segmented_files:
        file = []
        for s in i: # for each sentence in the file
            preprocessed_sentence = [token.text for token in s if not token.is_space and not token.is_stop and not token.is_punct]
            if len(preprocessed_sentence) != 0: # if the sentence is not empty
                file.append(preprocessed_sentence)
        preprocessed_segmented_files.append(file)
    return preprocessed_segmented_files

def tokenize(spacy_files):
    """
    Tokenizes the text data by removing stop words, punctuation, and spaces.
    """
    tokenized_files = []
    for i in spacy_files:
        tokenized_files.append([token.text for token in i if not token.is_space and not token.is_stop and not token.is_punct])
    return tokenized_files

def get_word_counts(tokenized_files: list)->dict:
    """
    Gets the word counts for the text data.
    """
    words = [token for file in tokenized_files for token in file]
    word_count = dict()
    for word in words:
        word_count[word] = word_count.get(word,0) + 1
    return {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}

def preprocess_text(texts:list, spacy_model:str='en_core_web_sm')->tuple:
    """
    Preprocesses the text data into spaCy doc and returns a list of spaCy docs, segmented sentences, tokenized words, and word counts.
    """
    nlp = spacy.load(spacy_model)
    spacy_text = get_spacy_text(texts,nlp)
    spacy_text_sents = sentencize(spacy_text)
    spacy_text_tokens = tokenize(spacy_text)
    spacy_text_wordcounts = get_word_counts(spacy_text_tokens)
    return spacy_text, spacy_text_sents, spacy_text_tokens, spacy_text_wordcounts

##### Analysis and Visualization #####

def create_word_cloud(word_counts, title):
    """
    Creates a word cloud visualization for the text data.
    """
    # print(category + ':')
    wordcloud = WordCloud(width=800, height=400, max_words=None, min_font_size=0.01, max_font_size=100,
                      background_color='white', colormap='gist_earth', random_state=42,
                      prefer_horizontal=1.0,
                      stopwords = []).generate_from_frequencies(word_counts)
    
    plt.figure(figsize=(16,6))
    plt.axis("off")
    plt.imshow(wordcloud, interpolation='bilinear')
    # plt.show()
    plt.savefig(f'analysis_results/{title}_wordcloud.png')
    print(f"***Word cloud for {title} saved in analysis_results/{title}_wordcloud.png***")

def get_sentence_stats(data_sents, category):
    """
    Computes the statistics for the text data. (Min, Max, Avg number of sentences, Histogram, Boxplot)
    """
    sent_counts = [len(file) for file in data_sents]
    min_sents = np.min(sent_counts)
    max_sents = np.max(sent_counts)
    avg_sents = np.mean(sent_counts)

    with open(f'analysis_results/{category}_sentence_stats.txt', 'w', encoding="utf-8") as f:
        print(f'Statistics for {category}', file=f)
        print(f'- Minimum number of sentences: {min_sents}', file=f)
        print(f'- Maximum number of sentences: {max_sents}', file=f)
        print(f'- Average number of sentences: {avg_sents}', file=f)
    print(f"***Sentence statistics for {category} saved in analysis_results/{category}_sentence_stats.txt***")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(sent_counts, bins=range(1, max_sents + 2), alpha=0.5, edgecolor='black')
    plt.xlabel('Number of sentences')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of sentence counts for {category}')

    plt.subplot(1, 2, 2)
    plt.boxplot(sent_counts)
    plt.xlabel('Number of sentences')
    plt.title(f'Box plot of sentence counts for {category}')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'analysis_results/{category}_sentence_stats.png')
    print(f"***Sentence statistics for {category} saved in analysis_results/{category}_sentence_stats.png***")

def count_bigrams(data_sents):
    """
    Counts the bigrams for the text data.
    """
    bigram_counter = Counter()
    bigrams_by_sent = []

    for file in data_sents:
        for sent in file:
            bigrams = list(nltk.bigrams(sent))
            bigram_counter.update(bigrams)
            bigrams_by_sent.append(len(bigrams))
    
    return bigram_counter, bigrams_by_sent

def compute_stats(bigrams):
    """
    Computes the statistics for the bigrams. (Min, Max, Avg number of bigrams)
    """
    min_bigrams = np.min(bigrams)
    max_bigrams = np.max(bigrams)
    avg_bigrams = np.mean(bigrams)
    return min_bigrams, max_bigrams, avg_bigrams

def get_bigram_stats(category1_sents, category2_sents, categories=['Computer Scientists', 'Sculptors']):
    """
    Computes the statistics for the bigrams. (Total, Min, Max, Avg number of bigrams, Bar plot)
    """
    bigram_counter_1, bigrams_1 = count_bigrams(category1_sents)
    bigram_counter_2, bigrams_2 = count_bigrams(category2_sents)

    total_bigrams_1 = sum(bigram_counter_1.values())
    total_bigrams_2 = sum(bigram_counter_2.values())

    min_bigrams_1, max_bigrams_1, avg_bigrams_1 = compute_stats(bigrams_1)
    min_bigrams_2, max_bigrams_2, avg_bigrams_2 = compute_stats(bigrams_2)

    total_bigrams = [total_bigrams_1, total_bigrams_2]
    min_bigrams = [min_bigrams_1, min_bigrams_2]
    max_bigrams = [max_bigrams_1, max_bigrams_2]
    avg_bigrams = [avg_bigrams_1, avg_bigrams_2]

    with open('analysis_results/bigram_stats.txt', 'w', encoding="utf-8") as f:
        print(f'Statistics for {categories[0]}', file=f)
        print(f'- Total number of bigrams: {total_bigrams[0]}', file=f)
        print(f'- Minimum number of bigrams: {min_bigrams[0]}', file=f)
        print(f'- Maximum number of bigrams: {max_bigrams[0]}', file=f)
        print(f'- Average number of bigrams: {avg_bigrams[0]}', file=f)
        print('*****************', file=f)
        print(f'Statistics for {categories[1]}', file=f)
        print(f'- Total number of bigrams: {total_bigrams[1]}', file=f)
        print(f'- Minimum number of bigrams: {min_bigrams[1]}', file=f)
        print(f'- Maximum number of bigrams: {max_bigrams[1]}', file=f)
        print(f'- Average number of bigrams: {avg_bigrams[1]}', file=f)
    print("***Bigram statistics saved in analysis_results/bigram_stats.txt***")

    _ , ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(categories, total_bigrams, color=['steelblue', 'purple'])
    ax[0].set_title('Total number of bigrams')
    ax[0].set_ylabel('Bigrams')

    x = np.arange(len(categories))
    width = 0.3

    ax[1].bar(x - width, min_bigrams, width, label='Min')
    ax[1].bar(x, max_bigrams, width, label='Max')
    ax[1].bar(x + width, avg_bigrams, width, label='Avg')

    ax[1].set_title('Number of bigrams per sentence')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(categories)
    ax[1].legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig('analysis_results/bigram_stats.png')
    print("***Visualisation for bigram statistics saved in analysis_results/bigram_stats.png***")
