import spacy
import stanza
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_spacy_ne(text, nlp_spacy):
    """
    Get named entities using Spacy
    """
    doc = nlp_spacy(text)
    ents = [(ent.text, ent.label_) for ent in tqdm(doc.ents, desc='Spacy Processing')]
    return ents

def get_stanza_ne(text, nlp_stanza):
    """
    Get named entities using Stanza
    """
    doc = nlp_stanza(text)
    ents = []
    for sent in tqdm(doc.sentences, desc='Stanza Processing'):
        for ent in sent.ents:
            ents.append((ent.text, ent.type))
    return ents

def get_ne_comparisons(spacy_ne, stanza_ne):
    """
    Compare the named entities extracted by Spacy and Stanza
    """
    spacy_dict = dict()
    stanza_dict = dict()
    
    for ne, label in spacy_ne:
         # If the key is not present, add it to the dictionary with an empty list as the value
        spacy_dict.setdefault(ne, []).append(label)

    for ne, label in stanza_ne:
        # Same as above
        stanza_dict.setdefault(ne, []).append(label)

    full_agree = 0
    full_agree_with_type_agree = 0
    partial_agree = 0
    partial_agree_with_type_agree = 0
    only_spacy = 0
    only_stanza = 0

    all_ne_predictions = spacy_dict.keys() | stanza_dict.keys()

    # Compare the named entities
    for ne in all_ne_predictions:
        if ne in spacy_dict.keys() and ne in stanza_dict.keys():
            full_agree += 1
            if spacy_dict[ne] == stanza_dict[ne]:
                full_agree_with_type_agree += 1
        elif ne in spacy_dict.keys():
            partial_matches = {k: v for k, v in stanza_dict.items() if k.startswith(ne)}
            if partial_matches:
                partial_agree += 1
                if any(spacy_dict[ne] == stanza_dict[k] for k in partial_matches.keys()):
                    partial_agree_with_type_agree += 1
            else:
                only_spacy += 1
        else:
            partial_matches = {k: v for k, v in spacy_dict.items() if k.startswith(ne)}
            if partial_matches:
                partial_agree += 1
                if any(stanza_dict[ne] == spacy_dict[k] for k in partial_matches.keys()):
                    partial_agree_with_type_agree += 1
            else:
                only_stanza += 1

    stats = {
        'Full agreement': full_agree,
        'Full agreement with the same NE type': full_agree_with_type_agree,
        'Partial agreement': partial_agree,
        'Partial agreement with the same NE type': partial_agree_with_type_agree,
        'Only predicted by Spacy': only_spacy,
        'Only predicted by Stanza': only_stanza
    }
    
    return stats

def get_ne_agreement_stats(category1_data, category2_data, categories = ['Computer Scientists', 'Sculptors']):
    """
    Get the statistics of named entity agreement between Spacy and Stanza
    """
    nlp_spacy = spacy.load("en_core_web_sm")
    nlp_stanza = stanza.Pipeline(lang='en', 
                                 processors='tokenize,ner',
                                 tokenize_pretokenized=True,
                                 logging_level='FATAL')
    
    #initialize the statistics
    all_stats = {categories[0]:
                 {'Full agreement': 0,
                  'Full agreement with the same NE type': 0,
                  'Partial agreement': 0,
                  'Partial agreement with the same NE type': 0,
                  'Only predicted by Spacy': 0,
                  'Only predicted by Stanza': 0},
                  categories[1]:
                  {'Full agreement': 0,
                   'Full agreement with the same NE type': 0,
                   'Partial agreement': 0,
                   'Partial agreement with the same NE type': 0,
                   'Only predicted by Spacy': 0,
                   'Only predicted by Stanza': 0}
                }
    
    all_texts = {categories[0]: category1_data,
                      categories[1]: category2_data}
    
    # Compare base on the category
    for category, texts in all_texts.items():
        combined_text = "\n".join(texts)

        spacy_ne = get_spacy_ne(combined_text, nlp_spacy)
        stanza_ne = get_stanza_ne(combined_text, nlp_stanza)
        
        stats = get_ne_comparisons(spacy_ne, stanza_ne)
        
        for k in all_stats[category]:
            all_stats[category][k] += stats[k]

    with open("analysis_results/ner_agreement_stats.txt", "w") as f:
        for k, v in all_stats.items():
            print('\n' + k, file=f)
            for k2,v2 in v.items():
                print(k2, ":", v2, file=f)

    print('*** statistics saved in ner_agreement_stats.txt ***')

    return all_stats



def visualize_agreement_stats(stats, categories = ['Computer Scientists', 'Sculptors']):
    """
    Visualize the statistics of named entity agreement between Spacy and Stanza
    """
    stats_keys = list(stats[categories[0]].keys())
    
    _, ax = plt.subplots(figsize=(12, 6))
    w = 0.3
    index = range(len(stats_keys))
    bars = [list(map(lambda x: x + w * i, index)) for i in range(2)]

    for i, category in enumerate(categories):
        stat_values = [stats[category][k] for k in stats_keys]
        ax.bar(bars[i], stat_values, w, label = category)

    ax.set_xlabel('Statistics')
    ax.set_ylabel('Count')
    ax.set_title('NER Agreement Statistics by Category')
    ax.set_xticks([c + w/2 for c in range(len(stats_keys))])
    ax.set_xticklabels(['Full agreement', 
                        'Full agreement\n with the same NE type',
                        'Partial agremeent', 
                        'Partial agreement\n with the same NE Type',
    'Only predicted by Spacy', 'Only predicted by Stanza'])
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("analysis_results/ner_agreement_stats.png")
    print('*** visualisation saved in ner_agreement_stats.png ***')