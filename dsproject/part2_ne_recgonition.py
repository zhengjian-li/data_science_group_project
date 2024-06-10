import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
import stanza
from tqdm import tqdm

def get_NEs(text, spacy_nlp, stanza_nlp):
    # Use spacy to tokenize the text
    spacy_doc = spacy_nlp(text)
    spacy_tokens = [[token.text for token in sent] for sent in spacy_doc.sents]
    stanza_doc = stanza_nlp(spacy_tokens) # Use pre-tokenized text for stanza

    # (NE, type) pairs
    spacy_NEs = [(ne.text, ne.label_) for ne in spacy_doc.ents]
    stanza_NEs = [(ne.text, ne.type) for sent in stanza_doc.sentences for ne in sent.ents]

    return spacy_NEs, stanza_NEs

def get_NEs_df(text_cs_files_name, text_sculpt_files_name, text_cs_files, text_sculpt_files):
    
    # Create a dictionary to store the NEs
    NEs = {'Category':[], 
       'Recognizer':[],
       'NE':[],
       'Type':[],
       'File':[]}
    
    spacy_nlp = spacy.load("en_core_web_sm")
    stanza_nlp = stanza.Pipeline(lang='en', 
                                 processors='tokenize,ner',
                                 tokenize_pretokenized=True,
                                 logging_level='FATAL')
    # In sculptors
    for name, text in tqdm(zip(text_sculpt_files_name, text_sculpt_files),
                           desc='Sculptors NER', 
                           total=len(text_sculpt_files_name)):
        spacy_NEs, stanza_NEs = get_NEs(text, spacy_nlp, stanza_nlp)

        for ne in spacy_NEs:
            NEs['Category'].append('Sculptors')
            NEs['Recognizer'].append('spacy')
            NEs['NE'].append(ne[0])
            NEs['Type'].append(ne[1])
            NEs['File'].append(name)
        
        for ne in stanza_NEs:
            NEs['Category'].append('Sculptors')
            NEs['Recognizer'].append('stanza')
            NEs['NE'].append(ne[0])
            NEs['Type'].append(ne[1])
            NEs['File'].append(name)

    # In Computer Scientists
    for name, text in tqdm(zip(text_cs_files_name, text_cs_files),
                           desc='Computer Scientists NER', 
                           total=len(text_cs_files_name)):
        spacy_NEs, stanza_NEs = get_NEs(text, spacy_nlp, stanza_nlp)

        for ne in spacy_NEs:
            NEs['Category'].append('Computer Scientists')
            NEs['Recognizer'].append('spacy')
            NEs['NE'].append(ne[0])
            NEs['Type'].append(ne[1])
            NEs['File'].append(name)
        
        for ne in stanza_NEs:
            NEs['Category'].append('Computer Scientists')
            NEs['Recognizer'].append('stanza')
            NEs['NE'].append(ne[0])
            NEs['Type'].append(ne[1])
            NEs['File'].append(name)

    NEs_df = pd.DataFrame(NEs)
    NEs_df.to_csv("data/part2/NEs.csv", index=False)
    print('*** NEs.csv has been saved ***')

def nes_visualisation(df: pd.DataFrame, title: str):
    """
    Visualize the statistics of NEs
    """
    fig, ax = plt.subplots(1,3, figsize=(12,3))
    fig.subplots_adjust(wspace=0.3)
    plt.figure(figsize=(5,5))

    sns.barplot(x='Category', y='mean', hue='Recognizer', data=df, ax=ax[0])
    sns.move_legend(
        ax[0], "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    sns.barplot(x='Category', y='min', hue='Recognizer', data=df, ax=ax[1])
    sns.move_legend(
        ax[1], "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )
    sns.barplot(x='Category', y='max', hue='Recognizer', data=df, ax=ax[2])
    sns.move_legend(
        ax[2], "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )

    plt.savefig(f"analysis_results/NEs_stats_c_{title}.png")
    print(f'*** visualisation to {title} saved in NEs_stats_c_{title}.png ***')

def nes_stats_visualisation(NEs_df):
    """
    To get the statistics of NEs and visualize them
    """
    # Get the number of NEs grouped by Category, Recognizer, and File
    NEs_nb = NEs_df.groupby(['Category','Recognizer','File']).size().reset_index(name='NE_nb')

    # Get the avg/min/max number of NEs
    NEs_a = NEs_nb.groupby(['Category','Recognizer'])['NE_nb'].agg(['mean','min','max']).reset_index()
    NEs_a.to_csv("analysis_results/NEs_stats_a.csv", index=False)
    print('*** avg/min/max number of NEs has been saved in NEs_stats_a.csv ***')

    # Get the avg/min/max number of words in each NE
    NEs_df['Words_nb'] = NEs_df['NE'].apply(lambda x: len(x.split())) # Get the number of words in each NE
    NEs_b = NEs_df.groupby(['Category','Recognizer'])['Words_nb'].agg(['mean','min','max']).reset_index()
    NEs_b.to_csv("analysis_results/NEs_stats_b.csv", index=False)
    print('*** avg/min/max number of words in each NE has been saved in NEs_stats_b.csv ***')

    nes_visualisation(NEs_a, 'n of NEs')
    nes_visualisation(NEs_b, 'n of Words in each NE')
