# Data Science Group Project

## Group Members
- Zhengjian Li
- Ezgi Başar 
- Arary Kiros Hailemariam

## Set up

**Requirement set up**

Download the folder where this README.md is, and install the required packages:

```bash
pip install -r requirements.txt
```

**Running the spaCy Model**

To download and set up the spaCy model, execute the following command:

```bash
python -m spacy download en_core_web_sm
```

**Running the Stanza Model**

To download and set up the Stanza model, use the following Python command:

```python
import stanza
stanza.download('en')
```
**Runing the NLTK moduel**

To download and set up the NLTK moduel, use the following Python command:

```python
import nltk
nltk.download('stopword')
nltk.download('punkt')
```

## Usage Instructions

### 1.1 Data Collection

To initiate the data collection process, execute the following command:

```bash
python 1_collect_data.py
```

By default, this script is configured to collect data in English and includes two categories. You can adjust parameters according to your specific requirements:

```bash
python 1_collect_data.py --category <category> --language <language> --data_limit <n> --language <la>
```

For the `--category` argument, permissible values include `sculptor`, `computer_scientist`, or `all`.

For the `--data_limit`, it accept integers.

For the `--language` argument, permissible you can use the language code.

The data will be saved in `data/part1`.

### 1.2 Named Entity Recognition: analysis by entity type

For getting the tatistics and visualisations from the questions, you can run the following code, or you can also run with parameters accordingly

```bash
python 1_data_analysis.py --vocabulary --sentences --tokens --rdf_properties --facts
```
### 1.2 Data Analysis and Visualization

To generate statistics and visualizations from the collected data, execute the following command:

```bash
python 1_data_analysis.py --vocabulary --sentences --tokens --rdf_properties --facts
```

You may also run the script with specific parameters as needed to tailor the analysis. The available parameters are:

- `--vocabulary`
- `--sentences`
- `--tokens`
- `--rdf_properties`
- `--facts`

## 1.3 Clustering

For getting the clustering statistics and visualization, you need to run:

```bash
python 1_clustering.py
```

## 2.1 Named Entity Recognition

For getting the named entity recognition statistics and visualization, you need to run:

```bash
python 2_ner.py
```

## 2.2 Named Entity Recognition: analysis by entity type

For getting the statistics and visualization of this task, you need to run:

```bash
python 2_analysis.py
```

## 2.3 Named Entity Recognition: verification against knowledge graph

For getting the statistics and visualization of this task, you need to run:

```bash
python 2_analysis_kg.py
```

