import pandas as pd
from dsproject.part2_ne_analysis_kg import calculate_ratio
from dsproject.part2_ne_analysis_kg import get_count_matching_results

if __name__ == '__main__':
    # Load the NEs and KG entities
    df_text = pd.read_csv('data/part2/NEs.csv')
    df_stanza = df_text[df_text['Recognizer'] == 'stanza'] # Keep only the Stanza NEs
    df_spacy = df_text[df_text['Recognizer'] != 'stanza'] # Keep only the SpaCy NEs

    df_results_stanza = get_count_matching_results(df_stanza, 'stanza')
    df_results_spacy = get_count_matching_results(df_spacy, 'spacy')

    # Calculate the ratio for Stanza
    ratio_stanza = calculate_ratio(df_results_stanza, df_stanza)

    # Calculate the ratio for SpaCy
    ratio_spacy = calculate_ratio(df_results_spacy, df_spacy)

    # Output the ratios
    with open('analysis_results/ratios_of_matching.txt', 'w', encoding='utf-8') as f:
        print(f"Ratio of matching NEs with KG for Stanza: {ratio_stanza:.2f}", file=f)
        print(f"Ratio of matching NEs with KG for SpaCy: {ratio_spacy:.2f}", file=f)
    print('*** Ratios have been saved in ratios_of_matching.txt ***')
