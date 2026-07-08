#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T19:19:18.272840Z","iopub.execute_input":"2026-06-29T19:19:18.273280Z","iopub.status.idle":"2026-06-29T20:10:07.347855Z","shell.execute_reply.started":"2026-06-29T19:19:18.273250Z","shell.execute_reply":"2026-06-29T20:10:07.347192Z"},"jupyter":{"outputs_hidden":false}}
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

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import wandb
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
wandb_api_key = user_secrets.get_secret("WANB_API_KEY")


wandb.login(
    key=wandb_api_key
)

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

                "label": 1 if option == row["answer"] else 0,

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
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.prompts[idx],
            self.options[idx],
            truncation=True,
            max_length=MAX_LENGTH
        )

        return {
            "input_ids":
                encoding['input_ids'],

            "attention_mask":
                encoding['attention_mask'],

            "labels": int(self.labels[idx])
        }

# ____________________________________________________________________________________________________
# 3.4 - process data to prepare it for training
# takes in a dataframe and returns dataloader for training
def process_data(dataframe, tokenizer, shuffle):
    expanded_df = expand_dataset(dataframe)
    print(expanded_df["label"].value_counts())
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
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return data_loader

# ____________________________________________________________________________________________________
# 3.5 - the actual training loop
def train_model(
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs,
    save_dir
):
    lowest_val_loss = float("inf")
    best_accuracy = 0

    for epoch in range(epochs):

        # Training - put model into training mode
        model.train()

        train_loss = 0

        for batch in train_loader:
            # basically move the data to gpu for computation
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # to convert them to floats32 for single precision calculations
            labels = batch["labels"].long().to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation
        model.eval()

        val_loss = 0

        correct = 0

        total = 0

        with torch.no_grad():
            for batch in val_loader:
                # basically move the data to gpu for computation
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # to convert them to floats32 for single precision calculations
                labels = batch["labels"].long().to(device)

                outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels=labels
                )

                val_loss += outputs.loss.item()

                predictions = torch.argmax(
                    outputs.logits,
                    dim=1
                )

                correct += (predictions == labels).sum().item()

                total += labels.size(0)

        val_loss /= len(val_loader)

        accuracy = correct/total

        ##################################
        # Save Best Model
        ##################################
        if val_loss < lowest_val_loss:
            best_accuracy = accuracy

            lowest_val_loss = val_loss

            model.save_pretrained(save_dir)

            tokenizer.save_pretrained(save_dir)

            artifact = wandb.Artifact(

                save_dir,

                type="model"

            )

            artifact.add_dir(save_dir)

            wandb.log_artifact(artifact)


         ##################################
        # W&B Logging
        ##################################

        wandb.log({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "validation_loss": val_loss,

            "validation_accuracy": accuracy,

            "learning_rate": scheduler.get_last_lr()[0],

            "lowest validation loss": lowest_val_loss

        })

        ##################################
        # Console
        ##################################

        print(

            f"Epoch {epoch+1}/{epochs}"

            f" | Train Loss: {train_loss:.4f}"

            f" | Val Loss: {val_loss:.4f}"

            f" | Accuracy: {accuracy:.4f}"

        )

    print(f"\nLowest Validation Loss: {lowest_val_loss:.4f}")

    return {
        'loss': lowest_val_loss,
        'accuracy': best_accuracy,
        'checkpoint': save_dir
    }

# 3.6 - clean up between stages and folds
def clear_data(model, optimizer, scheduler):
    try:
        del model
        del optimizer
        del scheduler

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        return("could not clean up model, optimizer and scheduler", e)

    return

def upload_models():
    for fold in range(1, (FOLDS+1)):
        kagglehub.model_upload(
            handle=f"siddharthavarma12/deberta-v3-base/fold{fold}-stage1",
            local_model_dir=f"best_model_fold{fold}_stage1"
        )

        kagglehub.model_upload(
            handle=f"siddharthavarma12/deberta-v3-base/fold{fold}-stage2",
            local_model_dir=f"best_model_fold{fold}_stage2"
        )

# ____________________________________________________________________________________________________
# ____________________________________________________________________________________________________
# ____________________________________________________________________________________________________

'''    kf = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED
    )

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(df, df['answer']),
        start=1
    ):
        print(f"\n========== Fold {fold} ==========\n")'''

'''    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(len(train_df), len(val_df))
    print(train_df["answer"].value_counts())
    print(val_df["answer"].value_counts())'''

def main():
    fold=1
    df = pd.read_csv("/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        dtype=torch.float32
    )

    model = model.to(device)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['answer'],
        random_state=42
    )

    train_loader = process_data(
        train_df,
        tokenizer,
        True
    )

    val_loader = process_data(
        val_df,
        tokenizer,
        False
    )

    # Stage - 1 
    # Setup
    # optimizer
    optimizer1 = AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    print(optimizer1.param_groups[0]["lr"])

    total_steps = len(train_loader) * STAGE1_EPOCHS

    # scheduler
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=int(0.1 * (total_steps)),
        num_training_steps=(total_steps)
    )

    print(total_steps)
    print(len(train_loader))
    print(STAGE1_EPOCHS)

    wandb.init(

        project="24f1002705-DLGenAI-t12026",

        name=f"deberta-v3-base-fold{fold}_stage1",

        config={

            "model": MODEL_NAME,
            "epochs": STAGE1_EPOCHS,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "max_length": MAX_LENGTH,
            "optimizer": "AdamW",
            "scheduler": "Linear Warmup"

        }

    )

    # trainer
    train_model(
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer1,
        scheduler1,
        STAGE1_EPOCHS,
        f"best_model_fold{fold}_stage1"
    )

    wandb.finish()

    # interstage clean up
    clear_data(model, optimizer1, scheduler1)


    # Stage - 2
    # setup
    # import model
    model = AutoModelForSequenceClassification.from_pretrained(
        f"best_model_fold{fold}_stage1"
    )

    model = model.to(device)

    optimizer2 = AdamW(
        model.parameters(),
        lr=5e-6,
        weight_decay=0.01
    )

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=int(0.1 * (len(train_loader) * STAGE2_EPOCHS)),
        num_training_steps=(len(train_loader) * STAGE2_EPOCHS)
    )

    wandb.init(

        project="24f1002705-DLGenAI-t12026",

        name=f"deberta-v3-base-fold{fold}_stage2",

        config={

            "model": MODEL_NAME,
            "epochs": STAGE2_EPOCHS,
            "batch_size": 16,
            "learning_rate": 5e-6,
            "max_length": MAX_LENGTH,
            "optimizer": "AdamW",
            "scheduler": "Linear Warmup"

        }

    )

    train_model(
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer2,
        scheduler2,
        STAGE2_EPOCHS,
        f"best_model_fold{fold}_stage2"
    )
    wandb.finish()

    clear_data(model, optimizer2, scheduler2)

if __name__ == "__main__":

    main()

#upload_models()

# %% [code] {"execution":{"iopub.status.busy":"2026-06-29T20:12:38.155255Z","iopub.execute_input":"2026-06-29T20:12:38.155902Z","iopub.status.idle":"2026-06-29T20:12:38.162492Z","shell.execute_reply.started":"2026-06-29T20:12:38.155868Z","shell.execute_reply":"2026-06-29T20:12:38.161541Z"}}
import torch
import torch.nn.functional as F

OPTIONS = ["A", "B", "C", "D", "E"]

def predict_mcq(model, tokenizer, row, device):

    model.eval()

    scores = []

    with torch.no_grad():

        for option in OPTIONS:

            encoding = tokenizer(

                row["prompt"],
                row[option],

                truncation=True,
                padding="max_length",
                max_length=256,

                return_tensors="pt"

            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(

                input_ids=input_ids,
                attention_mask=attention_mask

            )

            probabilities = F.softmax(outputs.logits, dim=1)

            score = probabilities[0,1].item()

            scores.append(score)

    ranking = [

        option

        for _, option in sorted(

            zip(scores, OPTIONS),

            reverse=True

        )

    ]

    return ranking, scores

