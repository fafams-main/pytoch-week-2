import pytest
import os

from model.train import APSClassifier
from model.dataloader import train_loader, val_loader


def test_resume_training():

    model = APSClassifier()
    
    model.fit(train_loader,val_loader,random_state=42, epochs=5, resume_train=False)

    assert model.last_epoch == 4, "Training did not complete the expected number of epochs."
    
    resume_train = APSClassifier()
    resume_train.fit(train_loader,val_loader,random_state=42, epochs=10, resume_train=True)

    assert resume_train.last_epoch == 9