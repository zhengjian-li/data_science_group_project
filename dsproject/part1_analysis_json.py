from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from .part1_analysis_txt import create_word_cloud #reuse the word cloud function

def get_top_fifty_properties_wordcloud(data, category):
    """
    Get the top 50 RDF properties and create a word cloud
    """
    predicate_counter = Counter() # Counter to store the frequency of each predicate
    for f in data:
        # Extract the predicates from the query results
        predicates = [binding['predicate']['value'] for binding in f['results']['bindings']]
        predicate_counter.update(predicates)
    top_fifty=dict(predicate_counter.most_common(50)) # Get the top 50 predicates and their frequencies

    # Create a word cloud
    create_word_cloud(top_fifty, f'Top-50 RDF properties for {category}')
    with open(f'analysis_results/top_fifty_properties_{category}.txt', 'w') as f:
        print(*top_fifty, sep="\n", file=f)
    print(f'***Top-50 RDF properties for {category} saved in analysis_results/top_fifty_properties_{category}.txt***')

def get_number_of_facts(data):
    """
    get the number of facts(triples) in each query result
    """
    fact_counts = []

    for f in data:
        num_facts = len(f['results']['bindings'])
        fact_counts.append(num_facts)

    return fact_counts

def get_fact_stats(data, category):
    """
    Get the statistics of the number of facts in each query result
    """
    
    fact_counts = get_number_of_facts(data)

    min_facts = np.min(fact_counts)
    max_facts = np.max(fact_counts)
    avg_facts = np.mean(fact_counts)
    
    with open(f'analysis_results/fact_stats_{category}.txt', 'w') as f:
        print(f'Statistics for {category}')
        print(f'- Minimum number of facts: {min_facts}', file=f)
        print(f'- Maximum number of facts: {max_facts}', file=f)
        print(f'- Average number of facts: {avg_facts}', file=f)
    print(f'***Statistics for {category} saved in analysis_results/fact_stats_{category}.txt***')

    # Visualize the statistics
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(fact_counts, bins=20, alpha=0.5, edgecolor='black')
    plt.xlabel('Number of facts')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of fact counts for {category}')

    plt.subplot(1, 2, 2)
    plt.boxplot(fact_counts)
    plt.xlabel('Number of facts')
    plt.title(f'Box plot of fact counts for {category}')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'analysis_results/fact_stats_{category}.png')
    print(f'***Plots for {category} saved in analysis_results/fact_stats_{category}.png***')
