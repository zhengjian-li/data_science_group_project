from dsproject.part1_clustering import cluster_and_visualize
from dsproject.part1_clustering import extract_named_entities_and_relations_from_json_file
from dsproject.part1_clustering import preprocess_json
from dsproject.part1_clustering import preprocess_text
from dsproject.part1_clustering import vectorizers
from dsproject.utils import read_files
import numpy as np

if __name__ == '__main__':

    text_cs_files, text_sculpt_files, json_cs_files, json_sculpt_files = read_files()

    cleaned_data = [preprocess_text(text) for text in text_cs_files + text_sculpt_files]
    true_labels = [0] * len(text_cs_files) + [1] * len(text_sculpt_files)

    # word2vec_embeddings, glove_embeddings_matrix, tfidf_vectors = vectorizers(cleaned_data)
    word2vec_embeddings, tfidf_vectors = vectorizers(cleaned_data)

    # plt.figure(figsize=(16, 12))

    # Visualize Word2Vec results
    cluster_and_visualize(word2vec_embeddings, 'Word2Vec_Text', true_labels)

    # Visualize GloVe results
    # cluster_and_visualize(glove_embeddings_matrix, 'GloVe_Text', true_labels)

    
    # Visualize TF-IDF results
    cluster_and_visualize(tfidf_vectors, 'TF-IDF_Text', true_labels)

    cluster_and_visualize(np.concatenate((tfidf_vectors[:100,:],tfidf_vectors[-100:,:]),  axis=0),
                            'TF-IDF_Text_200', 
                            true_labels[:100]+true_labels[-100:])
    
    cluster_and_visualize(np.concatenate((tfidf_vectors[:200,:],tfidf_vectors[-200:,:]),  axis=0),
                            'TF-IDF_Text_400', 
                            true_labels[:200]+true_labels[-200:])
    
    cluster_and_visualize(np.concatenate((tfidf_vectors[:300,:],tfidf_vectors[-300:,:]),  axis=0),
                            'TF-IDF_Text_600', 
                            true_labels[:300]+true_labels[-300:])
    ##################

    cs_triples = [extract_named_entities_and_relations_from_json_file(file) for file in json_cs_files]
    sculpt_triples = [extract_named_entities_and_relations_from_json_file(file) for file in json_sculpt_files]

    # Convert the 'triplets' column into a list of strings
    triplets_list = cs_triples + sculpt_triples
    cleaned_data = [preprocess_json(data) for data in triplets_list]
    # Encode the true labels
    true_labels = [0] * len(cs_triples) + [1] * len(sculpt_triples)

    
    # word2vec_embeddings, glove_embeddings_matrix, tfidf_vectors= vectorizers(cleaned_data)
    word2vec_embeddings, tfidf_vectors= vectorizers(cleaned_data)

    # Visualize Word2Vec results
    cluster_and_visualize(word2vec_embeddings, 'Word2Vec_Json', true_labels)

    # Visualize GloVe results
    # cluster_and_visualize(glove_embeddings_matrix, 'GloVe_Json', true_labels)

    # Visualize TF-IDF results
    cluster_and_visualize(tfidf_vectors, 'TF-IDF_Json', true_labels)

    cluster_and_visualize(np.concatenate((tfidf_vectors[:100,:],tfidf_vectors[-100:,:]),  axis=0),
                            'TF-IDF_Json_200', 
                            true_labels[:100]+true_labels[-100:])
    
    cluster_and_visualize(np.concatenate((tfidf_vectors[:200,:],tfidf_vectors[-200:,:]),  axis=0),
                            'TF-IDF_Json_400', 
                            true_labels[:200]+true_labels[-200:])
    
    cluster_and_visualize(np.concatenate((tfidf_vectors[:300,:],tfidf_vectors[-300:,:]),  axis=0),
                            'TF-IDF_Json_600', 
                            true_labels[:300]+true_labels[-300:])