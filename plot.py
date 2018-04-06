import matplotlib.pyplot as plt
import pandas as pd

def plot(x_label, y_label, **kwargs):
    for file, label in kwargs:
        data = pd.read_json(file)
        print(data)

if __name__== '__main__':
    plot('epoch', 'f1 score', 'unet16/run_eval-tag-f1.json' 'validation')