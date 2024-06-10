from dsproject.part2_ne_analysis_entity_type import get_ne_agreement_stats
from dsproject.part2_ne_analysis_entity_type import visualize_agreement_stats
from dsproject.utils import read_files

if __name__ == '__main__':
    # Read the text files
    text_cs_files, text_sculpt_files, json_cs_files, json_sculpt_files = read_files()
    # Get the named entity agreement statistics
    agreement_stats = get_ne_agreement_stats(text_cs_files, text_sculpt_files)
    # Visualize the statistics
    visualize_agreement_stats(agreement_stats)
