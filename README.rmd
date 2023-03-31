# Advanced NLP - Take Home Exam

This repository contains the code and data for the Advanced NLP Take Home Exam: Creating a challenge dataset for SRL systems



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the necesarry packages.

```bash
pip install -r requirements.txt
```

## Usage

To evaluate all three models (Bert-SRL, bi-LSTM, and the model from the previous assignment), the model from the previous assignment needs to be downloaded from https://drive.google.com/file/d/1lpyYyNLNhN-2M_X1zT_D19e6XnhyoXDQ/view?usp=share_link and placed in the folder bert4srl/saved_models/test/EPOCH_8. Once this is done, run the following command to evaluate all three models:


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