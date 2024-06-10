import json
import pandas as pd
import re

def extract_named_entities_from_json_file(file_path):
    """
    To extract named entities from a JSON file
    """
    file_path = re.sub('.txt', '.json', file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return set()
    except json.JSONDecodeError as e:
        print(f"An error occurred while decoding {file_path}: {e}")
        return set()
    
    entities = set()
    # Extract the entity items 
    for binding in json_data['results']['bindings']:
        subject_uri = binding['subject']['value']
        object_uri = binding['object']['value']
        subject_entity_matches = subject_uri.split('/')[-1]
        object_entity_matches = object_uri.split('/')[-1]
        if subject_entity_matches:
            entities.add(subject_entity_matches)
        if object_entity_matches:
            entities.add(object_entity_matches)
    return entities

def count_matching_entities(ne_list, kg_entities):
    """
    Function to count matching entities and record unique matches
    """
    # Compile a regular expression pattern for each NE to search for matches
    patterns = [re.compile(re.escape(ne), re.IGNORECASE) for ne in ne_list]
    
    # Initialize a set to hold all unique matches
    matched_entities = set()
    
    # Iterate over the patterns and search for matches in the kg_entities
    for pattern in patterns:
        # Find all matches for the current pattern
        matches = pattern.findall(' '.join(kg_entities))
        # Add each match to the set for uniqueness
        for match in matches:
            matched_entities.add(match)
    
    # Count the number of unique matches
    matching_count = len(matched_entities)
    matched_entities_list = list(matched_entities)
    return matching_count, matched_entities_list

def get_count_matching_results(df, category, is_save=True):
    """
    To count the matching entities and save the results
    """
    # Keep only the 'NE' and 'File' columns
    df = df[['NE', 'File']]
    # Group by 'File' and aggregate 'NE' into a list
    df_unique_paths = df.groupby('File')['NE'].apply(list).reset_index(name='NE_list')
    # Create an empty DataFrame to store the results
    df_results = pd.DataFrame(columns=['File', 'Matched_Entities', 'Matched_Entities_Count'])
    # Iterate over each row in the Stanza DataFrame
    for index, row in df_unique_paths.iterrows():
        file_path = 'data/part1/'+row['File']
        ne_list = row['NE_list']
        kg_entities = extract_named_entities_from_json_file(file_path)
        
        count, matching = count_matching_entities(ne_list, kg_entities)
        df_results.loc[index] = {'File': file_path, 'Matched_Entities': matching, 'Matched_Entities_Count': count}
    
    if is_save:
        df_results.to_csv(f"analysis_results/matching_entities_{category}.csv", index=False)
        print('*** Matching entities saved in matching_entities_{category}.csv ***')

    return df_results

# Function to calculate the ratio of matching NEs
def calculate_ratio(df_results, df_original):
    """
    Calculate the ratio of matching NEs to the total number of NEs
    """
    total_matching = df_results['Matched_Entities_Count'].sum()
    total_nes = df_original['NE'].nunique()
    ratio = total_matching / total_nes if total_nes > 0 else 0
    return ratio
