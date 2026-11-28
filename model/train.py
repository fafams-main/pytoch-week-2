

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, confusion_matrix,average_precision_score
import os
import argparse
torch.manual_seed(69)

class APSClassifier(nn.Module):
    def __init__(self,):
        super(APSClassifier,self).__init__()
        self.w1 = nn.Linear(170,64)
        self.w2 = nn.Linear(64,2)
        self.a1 = nn.ReLU()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.epochs_no_improve = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.last_epoch = -1


    def forward(self,x):
        w1 = self.w1(x)
        a1 = self.a1(w1)
        w2 = self.w2(a1)
        return (w2)

    def fit(self, train_loader, val_loader, epochs= 120,resume_train=False,random_state=None):
        train_losses = []
        val_losses = []
        patience = 20
        checkpoint = f'best_model_{random_state}.pth'
        latest_checkpoint = f'latest_model_{random_state}.pth'

        if os.path.exists(latest_checkpoint) and resume_train:
            self.load_checkpoint(latest_checkpoint)
            print(f"Resuming training from epoch {self.last_epoch + 1}")

        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, epochs):
            self.train()
            train_loss = 0.0

            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                predictions = self(x_batch)
                batch_loss = self.loss_function(predictions, y_batch)
                train_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    predictions = self(x_batch)
                    
                    val_loss += self.loss_function(predictions, y_batch).item()

                average_val_loss = val_loss / len(val_loader)

                if average_val_loss < self.best_val_loss:

                    self.best_val_loss = average_val_loss
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0

                    checkpoint_data = {
                    "model_state" : self.state_dict(),
                    "optimizer_state" : self.optimizer.state_dict(),
                    "epoch" : epoch,
                    "best_epoch" : self.best_epoch,
                    "best_val_loss" : self.best_val_loss,
                    "epochs_no_improve" : self.epochs_no_improve
                    }

                    torch.save(checkpoint_data, checkpoint)


                else:
                    self.epochs_no_improve += 1
                self.last_epoch = epoch

                latest_checkpoint_data = {
                    "model_state" : self.state_dict(),
                    "optimizer_state" : self.optimizer.state_dict(),
                    "epoch" : epoch,
                    "best_epoch" : self.best_epoch,
                    "best_val_loss" : self.best_val_loss,
                    "epochs_no_improve" : self.epochs_no_improve
                    }


                torch.save(latest_checkpoint_data, latest_checkpoint)
                
                val_losses.append(average_val_loss)
                average_train_loss = train_loss / len(train_loader)
                train_losses.append(average_train_loss)

            if self.epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if epoch < 10 or epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}" )

        print(f"Best Val Loss: {self.best_val_loss:.4f} at Epoch {self.best_epoch+1}")

        return train_losses, val_losses, self.best_val_loss, self.best_epoch, epochs,epoch,average_train_loss, average_val_loss,
                
    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=True, map_location="cpu")
        self.epochs_no_improve = checkpoint["epochs_no_improve"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        self.last_epoch = checkpoint["epoch"]

        print("Saved epoch:", checkpoint["epoch"] + 1)
        print("Saved best validation loss:", checkpoint["best_val_loss"])
        print("Saved best epoch:", checkpoint["best_epoch"] + 1)

        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
