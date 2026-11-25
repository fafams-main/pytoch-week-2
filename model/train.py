import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from dataloader import train_loader, val_loader
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, confusion_matrix,average_precision_score
from data import preprocess_data

torch.manual_seed(42)

class APSClassifier(nn.Module):
    def __init__(self,):
        super(APSClassifier,self).__init__()
        self.w1 = nn.Linear(170,64)
        self.w2 = nn.Linear(64,2)
        self.a1 = nn.ReLU()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)


    def forward(self,x):
        w1 = self.w1(x)
        a1 = self.a1(w1)
        w2 = self.w2(a1)
        return (w2)

    def fit(self, train_loader, val_loader, epochs= 120):
        train_losses = []
        val_losses = []
        patience = 20
        epochs_no_improve = 0
        best_val_loss = float('inf')
        best_epoch = 0
        checkpoint = 'best_model.pth'
        for epoch in range(epochs):
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

                if average_val_loss < best_val_loss:

                    best_val_loss = average_val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0

                    checkpoint_data = {
                        "model_state" : self.state_dict(),
                        "optimizer_state" : self.optimizer.state_dict(),
                        "epoch" : epoch,
                        "best_val_loss" : best_val_loss
                    }

                    torch.save(checkpoint_data, checkpoint)
                else:
                    epochs_no_improve += 1

                val_losses.append(average_val_loss)
                average_train_loss = train_loss / len(train_loader)
                train_losses.append(average_train_loss)

            if epochs_no_improve >= patience:
                epochs_no_improve += 1
                print(f"Early stopping at epoch {epoch+1}")
                break

            if epoch < 10 or epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}" )

        self.load_checkpoint(checkpoint)

        print(f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch+1}")

        return train_losses, val_losses, best_val_loss, best_epoch, epochs,epoch,average_train_loss, average_val_loss,
                
    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=True, map_location="cpu")

        print("Saved epoch:", checkpoint["epoch"] + 1)
        print("Saved best validation loss:", checkpoint["best_val_loss"])

        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        