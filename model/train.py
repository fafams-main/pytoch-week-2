import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from dataloader import train_loader, val_loader

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

    def fit(self, train_loader, val_loader, epochs= 100):
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
                    torch.save(self.state_dict(), checkpoint)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break


            val_losses.append(average_val_loss)
            average_train_loss = train_loss / len(train_loader)
            train_losses.append(average_train_loss)
            self.load_state_dict(torch.load(checkpoint))

            if epoch < 10 or epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}" )

        return train_losses, val_losses, best_val_loss, best_epoch, epochs,epoch,average_train_loss, average_val_loss,
        
train_losses, val_losses, best_val_loss, best_epoch, epochs,epoch,average_train_loss, average_val_loss, = APSClassifier().fit(train_loader, val_loader, epochs=100)


print(f"Epoch {epoch+1}/{epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")
print(f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch+1}")
    
def plot_losses(train_losses, val_losses, best_epoch, save_path=None):
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, train_losses, label="Train loss")
    plt.plot(epochs_range, val_losses, label="Val loss")

    # mark the epoch where best_model.pth was saved
    plt.axvline(best_epoch + 1, color="gray", linestyle="--", label=f"Best epoch ({best_epoch + 1})")
    plt.scatter(best_epoch + 1, val_losses[best_epoch], color="red", zorder=5)

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training vs validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

plot_losses(train_losses, val_losses, best_epoch, save_path="loss_curve.png")

def plot_metric(values, best_epoch, name):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(values) + 1), values, label=name)
    plt.axvline(best_epoch + 1, color="gray", linestyle="--", label="Best-loss epoch")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    train_losses, val_losses, best_val_loss, best_epoch, epochs,epoch,average_train_loss, average_val_loss, = APSClassifier().fit(train_loader, val_loader, epochs=100)
    plot_losses(train_losses, val_losses, best_epoch, save_path="loss_curve.png")
    plot_metric(train_losses, best_epoch, "Train Loss")
    plot_metric(val_losses, best_epoch, "Validation Loss")

if __name__ == "__main__":
    main()