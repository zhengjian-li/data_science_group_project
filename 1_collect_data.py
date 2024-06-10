import argparse
from dsproject.part1_data_collection import DataCollectorConfig
from dsproject.part1_data_collection import get_bio_kg

def parse_args():
    parser = argparse.ArgumentParser(description='Data Collection')
    parser.add_argument('--category', type=str, default='all', help='Category of the person: sculptor, computer_scientist, or all')
    parser.add_argument('--data_limit', type=int, default=400, help='Data limit')
    parser.add_argument('--language', type=str, default='en', help='Language')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.category == 'all': # Collect data for both categories
        config = DataCollectorConfig(category='sculptor', data_limit=args.data_limit, language=args.language)
        get_bio_kg(config)
        config = DataCollectorConfig(category='computer_scientist', data_limit=args.data_limit, language=args.language)
        get_bio_kg(config)
    else: # Collect data for the specified category
        config = DataCollectorConfig(category=args.category, data_limit=args.data_limit, language=args.language)
        get_bio_kg(config)
