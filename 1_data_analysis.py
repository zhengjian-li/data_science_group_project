import argparse

from dsproject.part1_analysis_json import get_fact_stats
from dsproject.part1_analysis_json import get_top_fifty_properties_wordcloud
from dsproject.part1_analysis_txt import create_word_cloud
from dsproject.part1_analysis_txt import get_bigram_stats
from dsproject.part1_analysis_txt import get_sentence_stats
from dsproject.part1_analysis_txt import preprocess_text
from dsproject.utils import read_files

def parse_args():
    parser = argparse.ArgumentParser(description='Data Analysis')
    parser.add_argument('--data_path', type=str, default="data/part1", help='Path to the data files')
    parser.add_argument('--language', type=str, default='en', help='Language of the data files')
    parser.add_argument("--vocabulary", action="store_true", help="50 most frequent words and word cloud for each category (Text)")
    parser.add_argument("--sentences", action="store_true", help="Min/max/avg number of sentences per category together with the corresponding histograms and box plots")
    parser.add_argument("--tokens", action="store_true", help="Total number of bi-gram occurrences per category. Min/max/avg number of bi-gram occurrences per sentence per category.")
    parser.add_argument("--rdf_properties", action="store_true", help="50 most frequent RDF properties for each category (JSON)")
    parser.add_argument("--facts", action="store_true", help="Min/max/avg number of facts per category together with the corresponding histograms and box plots")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Read the data files
    text_cs_files, text_sculpt_files, json_cs_files, json_sculpt_files = read_files(args.data_path, args.language)

    # Preprocess the text data
    text_cs, text_sents_cs, text_tokens_cs, text_wordcounts_cs = preprocess_text(text_cs_files)
    text_sculpt, text_sents_sculpt, text_tokens_sculpt, text_wordcounts_sculpt = preprocess_text(text_sculpt_files)

    if args.vocabulary:
        # Create a word cloud for the computer scientists
        create_word_cloud(text_wordcounts_cs, "Computer Scientists")
        # Create a word cloud for the sculptors
        create_word_cloud(text_wordcounts_sculpt, "Sculptors")

    if args.sentences:
        # Get the sentence statistics for the computer scientists
        get_sentence_stats(text_sents_cs, "Computer Scientists")
        # Get the sentence statistics for the sculptors
        get_sentence_stats(text_sents_sculpt, "Sculptors")
        
    if args.tokens:
        # Get the bigram statistics
        get_bigram_stats(text_sents_cs, text_sents_sculpt)
    
    if args.rdf_properties:
        # Create a word cloud for the top 50 RDF properties for computer scientists
        get_top_fifty_properties_wordcloud(json_cs_files, "Computer Scientists")
        # Create a word cloud for the top 50 RDF properties for sculptors
        get_top_fifty_properties_wordcloud(json_sculpt_files, "Sculptors")

    if args.facts:
        # Get the fact statistics for computer scientists
        get_fact_stats(json_cs_files, "Computer Scientists")
        # Get the fact statistics for sculptors
        get_fact_stats(json_sculpt_files, "Sculptors")
