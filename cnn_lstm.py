
'''
This file implements the required functions to train a CNN+LSTM (ConvLSTM) utilizing time series of GOES images.
The PyTorch implementation of ConvLSMT can be found at https://github.com/ndrplz/ConvLSTM_pytorch.
'''

from convlstm import ConvLSTM 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np


#Main model
class FireConvLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.convlstm = ConvLSTM(
            input_dim=2,
            hidden_dim=32,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        self.pool = nn.AdaptiveAvgPool2d(1) #Average to reduce spatial dims (32,64,64) -> (32,1,1)

        self.fc = nn.Linear(32, 1) #1 value from 32 features

    def forward(self, x):

        # x: (batch, time, 2, 64, 64)
        layer_output, last_state = self.convlstm(x)
        # Last layer output
        # (batch, time, 32, 64, 64)
        out = layer_output[0] 
        # Select final time
        # (batch, 32, 64, 64)
        out = out[:, -1]
        # Global average pooling
        # (batch, 32, 1, 1)
        out = self.pool(out)
        # (batch, 32)
        out = out.flatten(1)
        # (batch, 1)
        out = self.fc(out)

        return out.squeeze(1)



def train_model(
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs:int,
        model:FireConvLSTM,
        criterion,
        optimizer,
        device
):
    for epoch in range(epochs):

        # -----------------
        # TRAIN
        # -----------------

        model.train()

        train_loss = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad() #remove gradientes from previous batch

            logits = model(X_batch) #Run batch through model

            loss = criterion(logits, y_batch)

            loss.backward() #Backpropagate

            optimizer.step() #Update wrights

            train_loss += loss.item()

        train_loss /= len(train_loader)


        # -----------------
        # VALIDATION
        # -----------------

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)

                loss = criterion(logits, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

    return epoch+1,train_loss,val_loss

def compute_metrics(
        model:FireConvLSTM,
        val_loader:DataLoader,
        device,
):
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.numpy())


    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = (all_probs >= 0.5).astype(int)


    
    cm = confusion_matrix(
        all_targets,
        all_preds
    )
    precision = precision_score(
    all_targets,
    all_preds
    )
    recall = recall_score(
    all_targets,
    all_preds
    
    )   
    f1 = f1_score(
    all_targets,
    all_preds
    )

    return cm, precision, recall,f1