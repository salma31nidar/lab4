# NLP Language Models with PyTorch

## Objective
The main objective of this lab is to gain familiarity with NLP language models using the PyTorch library.

## Part 1: Classification Regression

### 1. Data Collection
Using web scraping libraries (Scrapy/BeautifulSoup), I collected text data from several Arabic websites focused on health (siha). 
The score represents the relevance of each text, rated on a scale from 0 to 10.

### 2. Preprocessing NLP Pipeline
A preprocessing pipeline was established, which included the following steps:
- Tokenization
- Stemming
- Lemmatization
- Removal of stop words
- Discretization

### 3. Model Training
The following models were trained using the preprocessed dataset:
- RNN (Recurrent Neural Network)
- Bidirectional RNN
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)

Hyperparameters were tuned to achieve the best performance for each model.

### 4. Model Evaluation
The models were evaluated using standard metrics such as accuracy, loss, and F1 score. Additionally, the BLEU score was used to assess the performance.

**Best Model**: The LSTM model showed the best performance in terms of accuracy, F1 score, and BLEU score, making it the most effective model for this task.

## Part 2: Transformer (Text Generation)

### 1. Fine-Tuning GPT-2
The `pytorch-transformers` library was installed, and the pre-trained GPT-2 model was loaded. The model was fine-tuned using a dataset of children's books.

### 2. Text Generation
The fine-tuned model was used to generate a new story paragraph based on a given sentence.

## Part 3: BERT

### Data Preparation
The dataset from [Amazon reviews](https://nijianmo.github.io/amazon/index.html) was used.

### 1. Establishing the Model
The pre-trained `bert-base-uncased` model was established.

### 2. Adapting the BERT Embedding Layer
The data was prepared, and the BERT embedding layer was adapted to the dataset.

### 3. Fine-Tuning and Training
The model was fine-tuned and trained, with optimal hyperparameters chosen to achieve an efficient model.

### 4. Model Evaluation
The model was evaluated using standard metrics such as accuracy, loss, and F1 score. Additionally, the BLEU score and BERT-specific metrics were used.

## General Conclusion
The use of pre-trained BERT models significantly improves the performance of NLP tasks. Fine-tuning allows the model to adapt to specific datasets, resulting in better accuracy and relevance in text generation and classification tasks.
