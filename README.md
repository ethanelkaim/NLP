
# NLP and ML Projects Repository

Welcome to the NLP and ML Projects Repository. This repository contains the code and related files for two Natural Language Processing (NLP) projects. These projects involve various NLP tasks such as Part-of-Speech (POS) Tagging and Named Entity Recognition (NER) using different machine learning models.
## Project 1 : Part-Of-Speech Tagging
The first project involves implementing models for Part-Of-Speech (POS) Tagging. The task is to develop and evaluate models that can accurately predict the POS tags of words in a given sentence. The following models are implemented:

- Model 1 (Large Model): Uses a set of predefined features and is trained on a large dataset.

- Model 2 (Small Model): A simplified version with fewer features and trained on a smaller dataset.

### Files:
- train1.wtag: Training data for Model 1.
- test1.wtag: Test data for Model 1.
- train2.wtag: Training data for Model 2.
- prepocessing.py: A file containing the creation of the features and histories.
- optimization.py: A file containing what is required for optimization.
- inference.py: A file where the MEMM Viterbi algorithm is implemented.
- main.py: A file that runs the program and outputs the tagging file.

### Evaluation:
Models are evaluated based on accuracy.
## Project 2 : Named Entity Recognition
The second project focuses on Named Entity Recognition (NER). The task is to implement various models to identify and classify named entities in a text. The following models are implemented:

- Simple Linear Model: Using basic KNN model.
- Feed Forward Neural Network (FFNN): Utilizes word embeddings (GloVe).
- LSTM: Advanced model using Recurrent Neural Networks or Long Short-Term Memory networks.

### Files:
- train.tagged, dev.tagged, test.untagged: Datasets for training, development, and testing.
- generate_comp_tagged.py: Script for generating files.
- main2.py: A file where the data is preprocessed and the models are implemented.

### Evaluation:
Models are evaluated using the F1-score.
