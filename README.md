# Named Entity Recognition (NER) using Word2Vec and TensorFlow
## Overview
This project implements **Named Entity Recognition (NER)** using a deep learning-based approach. The dataset is preprocessed, tokenized, and embedded using Word2Vec, followed by training a sequence model using **TensorFlow.**

## Dataset
**We use the Entity Annotated Corpus dataset, specifically the ner_dataset.csv file. This dataset contains:**
- Sentence #: Identifies sentences in the dataset.
- Word: Individual words in the sentences. 
- POS: Part-of-Speech tags.
- Tag: Named Entity Recognition (NER) labels.

NER Tag	Count
O (Non-entity)	887,908
B-geo (Geographical Entity)	37,644
B-tim (Time Expression)	20,333
B-org (Organization)	20,143
I-per (Person)	17,251
B-per (Person)	16,990
....


## Project Steps
### 1. Data Preprocessing
- Load dataset (ner_dataset.csv).
- Handle missing values using forward-fill (ffill).
- Extract words and entity labels.
- Convert dataset into a list of tokenized sentences.

### 2. Word and Tag Indexing
- Create a word index (w_index) mapping words to unique integer values.

- Create a tag index (t_index) for NER labels.

### 3. Word Embeddings with Word2Vec
- Train a Word2Vec model (vector_size=100, window=5, min_count=1).

- Generate word embeddings to represent words as numerical vectors.

### 4. Data Preparation for Training
- Convert words and tags into indexed format.
- Apply padding to ensure uniform input sizes.
- Convert NER labels to one-hot encoding.

### 5. Train-Test Split
- 90% of data for training, 10% for testing.

### 6. Embedding Matrix Creation
Construct an embedding matrix that maps words to vector representations.

## Requirements
To run this project, install the following dependencies:

```bash
pip install numpy pandas matplotlib tensorflow gensim scikit-learn
```
## How to Run
### 1. Clone the repository:
```bash
git clone https://github.com/your-username/NER-Project.git
cd NER-Project
```
### 2. Run the Jupyter Notebook or Python script:
```bash
python ner_training.py
```
### 3. Train and evaluate the model.

## Future Work
- Implement BiLSTM-CRF for better entity recognition.
- Use pretrained embeddings (e.g., FastText, GloVe).
- Experiment with Transformer-based models (e.g., BERT, RoBERTa).

## Acknowledgments
- Dataset from Entity Annotated Corpus.
- Word embeddings generated using Word2Vec.
- Model trained using TensorFlow & Keras.
