import json
import re
import string

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split



def cluster_and_visualize(embeddings, title, true_labels):
    """
    # to perform clustering and visualization
    """
    # Split the data into training and test sets
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        embeddings, true_labels, test_size=0.2, random_state=42)

    # Train KMeans model on the training data
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    kmeans.fit(train_embeddings)

    # Predict clusters on the test data
    predicted_clusters_test = kmeans.predict(test_embeddings)

    # Evaluate clustering on the test data using silhouette score
    silhouette_test = silhouette_score(test_embeddings, predicted_clusters_test)

    # Evaluate clustering using supervised metrics
    ari_test = adjusted_rand_score(test_labels, predicted_clusters_test)
    fmi_test = fowlkes_mallows_score(test_labels, predicted_clusters_test)
    homogeneity_test = homogeneity_score(test_labels, predicted_clusters_test)

    # Print the evaluation metrics
    with open(f'analysis_results/kmeans_eval_results_{title}.txt', 'w') as f:
        print(f'Silhouette Score ({title}): {silhouette_test}', file=f)
        print(f'Adjusted Rand Index ({title}): {ari_test}', file=f)
        print(f'Fowlkes-Mallows Index ({title}): {fmi_test}', file=f)
        print(f'Homogeneity Score ({title}): {homogeneity_test}', file=f)
    print(f"***KMeans Evaluation Results for {title} saved in analysis_results/kmeans_eval_results_{title}.txt***")

    # Fit PCA on the training data and transform the test data
    pca = PCA(n_components=2)
    pca.fit(train_embeddings)
    test_embeddings_2d = pca.transform(test_embeddings)

    # Visualize the clusters of the test data
    _, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c=predicted_clusters_test, cmap='viridis')
    ax.set_title(f'KMeans Clusters ({title})')
    ax.set_xlabel(f'{title} Feature 1')
    ax.set_ylabel(f'{title} Feature 2')
    plt.tight_layout()
    plt.savefig(f'analysis_results/kmeans_clusters_{title}.png')
    print(f"***KMeans Clusters for {title} saved in analysis_results/kmeans_clusters_{title}.png***")

def preprocess_text(text: str) -> str:
    """
    To preprocess text data
    """
    # Remove links, punctuation, numbers, and stopwords
    text = re.sub(r"http\S+", "", text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = re.sub(r"\d+", '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
    text = " ".join(tokens).lower().strip()
    return text

# Function that computes different methods of vectorization
def vectorizers(cleaned_data):
    """
    To compute different methods of vectorization
    """
    # Tokenize the text data
    tokenized_data = [word_tokenize(text.lower()) for text in cleaned_data]

    # Generate Word2Vec embeddings
    word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=50, window=5, min_count=1, workers=4)
    word2vec_embeddings = np.array([np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv] or [np.zeros(50)], axis=0) for words in tokenized_data])

    ## The Glove embedding are too large to summit
    
    # # Load GloVe embeddings
    # glove_file_path = 'glove.6B/glove.6B.100d.txt'  # Replace with your actual GloVe file path
    # glove_embeddings = {}
    # with open(glove_file_path, 'r') as file:
    #     for line in file:
    #         values = line.split()
    #         word = values[0]
    #         vector = np.asarray(values[1:], dtype='float32')
    #         glove_embeddings[word] = vector
    # glove_embeddings_matrix = np.array([np.mean([glove_embeddings.get(word, np.zeros(100)) for word in text.split()], axis=0) for text in cleaned_data])

    # Generate TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer(max_features=50)  # Adjust the number of features as needed
    tfidf_vectors = tfidf_vectorizer.fit_transform(cleaned_data).toarray()
    
    # return word2vec_embeddings, glove_embeddings_matrix, tfidf_vectors
    return word2vec_embeddings, tfidf_vectors

def extract_named_entities_and_relations_from_json_file(file):
    """
    To extract named entities and relations from a JSON file
    """
    json_data = json.loads(json.dumps(file))
    triples = []
    for binding in json_data['results']['bindings']:
        subject_uri = binding['subject']['value']
        predicate_uri = binding['predicate']['value']
        object_uri = binding['object']['value']
        # Extract the entity labels from URIs using regular expressions
        subject_entity_matches = subject_uri.split('/')[-1]
        predicate_matches = predicate_uri.split('/')[-1]
        object_entity_matches = object_uri.split('/')[-1]
        triples.append((subject_entity_matches, predicate_matches, object_entity_matches))
    return str(triples)

def preprocess_json(text: str) -> str:
    """
    To preprocess json data
    """
    # Remove links, punctuation, numbers, and stopwords
    # text = re.sub(r"http\S+", "", text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # text = re.sub(r'[Qq]\d+', '', text)
    # text = re.sub(r"\d+", '', text)
    tokens = word_tokenize(text)
    # tokens = [w for w in tokens if not w.lower() in stopwords.words("english") and w != 'p']
    text = " ".join(tokens).lower().strip()
    return text
