
import sys
import os
#import shutil

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from adapter import SparkAdapter
from train import Trainer

trainer = Trainer()

def test_train_models():
    assert trainer.train_models('./data/generated.csv')
