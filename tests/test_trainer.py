
import sys
import os
#import shutil

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from adapter import SparkAdapter
from train import Trainer

trainer = Trainer()

# удаляем файлы моделей, если они есть
#model_files = ['TF_MODEL', 'IDF_FEATURES', 'IDF_MODEL', 'WATCHED']
#for filename in model_files:
#    path = os.path.join(os.getcwd(), 'models', filename)
#    if os.path.exists(path):
#        if os.path.isdir(path):
#            shutil.rmtree(path)
#        else:
#            os.remove(path)

def test_train_models():
    assert trainer.train_models('./data/generated.csv')
