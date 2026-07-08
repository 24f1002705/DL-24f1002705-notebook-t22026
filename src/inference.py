#!/usr/bin/env python
# coding: utf-8

# In[4]:


# %% [code] {"execution":{"iopub.status.busy":"2026-07-08T09:08:07.156473Z","iopub.execute_input":"2026-07-08T09:08:07.157172Z","iopub.status.idle":"2026-07-08T09:08:13.442200Z","shell.execute_reply.started":"2026-07-08T09:08:07.157136Z","shell.execute_reply":"2026-07-08T09:08:13.439836Z"},"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

# ____________________________________________________________________________________________________
# 1. imports
import os
import gc
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import kagglehub

from kaggle_secrets import UserSecretsClient

from sklearn.model_selection import StratifiedKFold

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# ____________________________________________________________________________________________________
# 2. Setup
#2.1 - downlaod competition
kagglehub.competition_download('smart-mcq-solver-challenge')

# 2.2 - initialize dataframe
# df = pd.read_csv("/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv")

# 2.3 - SEED (for repeatability)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 2.4 - Weights and Biases setup
user_secrets = UserSecretsClient()

#2.5 - Constants
BATCH_SIZE = 16
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
FOLDS = 5
MODEL_NAME = "microsoft/deberta-v3-base"
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 5
MAX_LENGTH = 256

# ____________________________________________________________________________________________________
# ____________________________________________________________________________________________________
# ____________________________________________________________________________________________________
# 3. Functions and Auxilary functional elements

#3.1 - MAP@3 function
# takes two inputs, one is a list of all correct answers, the other a list of corresponding predictions
def map_at_3(y_true, y_pred):
    score = 0.0

    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch")

    for i in range(len(y_true)):
        pred = y_pred[i]
        true = y_true[i]
        if true in pred:
            for j in range(len(pred)):
                if pred[j] == true:
                    score += 1/(j+1)
                    break

    return score/len(y_true)

# ____________________________________________________________________________________________________
# 3.2 - expands dataset 
# takes in a row and creates 5 corresponding rows for each option with whether it is correct or not.
def expand_dataset(df):
    samples = []

    for _, row in df.iterrows():

        for option in ["A","B","C","D","E"]:
            samples.append({
                'prompt': row['prompt'],

                'option': row[option],

                "question_id": row["id"]
            })

    return pd.DataFrame(samples)

# ____________________________________________________________________________________________________
# 3.3 - MCQDataset Class for making table into an object
# create MCQDATASET class
class MCQDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.prompts = dataframe['prompt'].tolist()
        self.options = dataframe['option'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.prompts[idx],
            self.options[idx],
            truncation=True,
            max_length=256
        )

        return {
            "input_ids":
                encoding['input_ids'],

            "attention_mask":
                encoding['attention_mask']
        }

# ____________________________________________________________________________________________________
# 3.4 - process data to prepare it for training
# takes in a dataframe and returns dataloader for training
def process_data(dataframe, tokenizer, shuffle):
    expanded_df = expand_dataset(dataframe)
    data_object = MCQDataset(expanded_df, tokenizer)

    # data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    data_loader = DataLoader(
        data_object,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=data_collator,
        pin_memory=True
    )

    return data_loader

# ____________________________________________________________________________________________________
# 3.6 - clean up between stages and folds
def clear_data(model):
    try:
        del model

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        return("could not clean up model")

    return

# __________________________________________________________________________________________________________
# 3.7 - get models
def get_model_paths():
    folds = [1,2,3,5]
    paths = []
    for fold in folds:
        paths.append(f"/kaggle/input/models/siddharthavarma12/smart-mcq-deberta/pytorch/fold{fold}/1/best_model_fold{fold}_stage2")
    return paths


# 3.8 - predict for one model
def predict_model(
    test_loader,
    model
):
    model.eval()

    logits = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask

            )

            positive_logits = outputs.logits[:,1]

            logits.extend(

                positive_logits.cpu().numpy()

            )

    logits = np.array(logits)

    logits = logits.reshape(-1,5)

    clear_data(model)

    return logits

def ensemble_predictions(
    test_loader
):
    predictions = []

    for path in get_model_paths():
        model = AutoModelForSequenceClassification.from_pretrained(
            path
        )
        model = model.to(device)
        pred = predict_model(test_loader, model)
        predictions.append(pred)

    predictions = np.stack(
        predictions,
        axis=0
    )

    predictions = np.mean(
        predictions,
        axis=0
    )

    return predictions

def create_submission(predictions, df):
    options = np.array([
        "A",

        "B",

        "C",

        "D",

        "E"]
    )

    final_predictions = []

    for row in predictions:
        ranking = np.argsort(
            -row
        )

        prediction = " ".join(

            options[ranking[:3]]

        )

        final_predictions.append(prediction)

    submission = pd.DataFrame({
        "id": df["id"],
        "prediction": final_predictions        
    })

    submission.to_csv(
        "submission.csv",
        index=False
    )

    return submission

# ____________________________________________________________________________________________________
# ____________________________________________________________________________________________________
# ____________________________________________________________________________________________________
def main():
    df = pd.read_csv("/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_loader = process_data(
        df,
        tokenizer,
        False
    )

    predictions = ensemble_predictions(test_loader)

    submission = create_submission(
        predictions,
        df
    )
    submission.head()

if __name__ == "__main__":

    main()

