from dsproject.part2_ne_recgonition import get_NEs_df, nes_stats_visualisation
from dsproject.utils import read_files
import pandas as pd

if __name__ == '__main__':
    # Read the text files and get the names
    text_cs_files_name, text_sculpt_files_name, _, _ = read_files(is_file_name=True)
    text_cs_files, text_sculpt_files, _, _ = read_files()
    # Get the NEs and save them to a file
    get_NEs_df(text_cs_files_name, text_sculpt_files_name, text_cs_files, text_sculpt_files)
    # Read the NEs from the file
    NEs_df= pd.read_csv("data/part2/NEs.csv")
    # Visualize the NEs statistics
    nes_stats_visualisation(NEs_df)