# Advanced NLP - Exam

This repository contains the code and data for the ANLP Exam: Creating a challenge dataset for SRL systems

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necesarry packages.

```bash
pip install -r requirements.txt
```

## Usage

To evaluate all three models (Bert-SRL, bi-LSTM, and the model from the previous assignment), the model from the previous assignment needs to be downloaded from https://drive.google.com/file/d/1HNUZHUFJcP7WBz_Ka90QyHsrLZY2ZSyI/view?usp=sharing and placed in the folder bert4srl/saved_models/test/EPOCH_8. Once this is done, run the following command to evaluate all three models:


```python
python main.py
```

If you cannot download the models, or if you only want to evaluate the AllenNLP models, run the following command:

```python
python main_only_allen.py
```

This will evalaute the two allen_nlp models
## data

The JSON files for the 9 tests that are evaluated in this repository can be found in the 'testdata/' folder.

## Log Files

The Log files directory contains both error and succes logs of the different models on the different test sets. It provides a clear overview of the perfomance without having to run the code