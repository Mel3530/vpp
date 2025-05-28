# vpp
Code for the "Pathogenicity prediction of genomic variants based on machine learning techniques" paper

## Execution

```bash
pip install -r requirements.txt
python main.py
```

## File structure

```
│
├── data/
│   ├── clinvitae_2017.csv       ← Training dataset
│   ├── clinvitae_2019.csv       ← Internal test dataset
│   └── icr639.csv               ← External validation dataset
│
├── src/
│   ├── preprocessing.py         ← Column cleaning 
│   ├── feature_selection.py     ← Feature selection using XGBoost (top features)
│   ├── models.py                ← Model definitions and hyperparameter search space
│   ├── train_pipeline.py        ← Training, cross-validation, and heuristic optimization
│   └── utils.py                 ← Evaluation metrics and heuristic scoring function
│
├── main.py                      ← Orchestrates the entire training and evaluation pipeline
├── requirements.txt             ← Python dependencies
└── results.csv                  ← Output: metrics summary for each model
```
