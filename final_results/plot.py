import matplotlib.pyplot as plt
import json
import os
import numpy as np

def clean_nans(data):
    return [0 if value == 'NaN' else value for value in data]

def plot(x_label, y_label, plot_args):
    for label, args in sorted(plot_args.items()):
        with open(args['path']) as data_file:    
            data = json.load(data_file)
            plt.plot([x[1] for x in data], clean_nans([y[2] for y in data]), label=label, color=args['color'], linestyle=args['line_style'])
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.yticks(np.arange(0, 1.1, .1))
    plt.ylabel(y_label)
    plt.show()


if __name__== '__main__':
    # plot_args = {'original network training': {'path': 'ugan16_max_pool/run_train-tag-f1.json', 'color': 'b', 'line_style': '--'},
    #             'original network validation': {'path': 'ugan16_max_pool/run_eval-tag-f1.json', 'color': 'b', 'line_style':'solid'},
    #             'no sparse gradient network training': {'path': 'ugan16_avg_pool/run_train-tag-f1.json', 'color': 'orange', 'line_style': '--'},
    #             'no sparse gradient network validation': {'path': 'ugan16_avg_pool/run_eval-tag-f1.json', 'color': 'orange', 'line_style': 'solid'}}

    plot_args = {'classical training': {'path': 'unet16/run_train-tag-f1.json', 'color': 'b', 'line_style': '--'},
            'classical validation': {'path': 'unet16/run_eval-tag-f1.json', 'color': 'b', 'line_style':'solid'},
            'adversarial training': {'path': 'ugan16_max_pool/run_train-tag-f1.json', 'color': 'orange', 'line_style': '--'},
            'adversarial validation': {'path': 'ugan16_max_pool/run_eval-tag-f1.json', 'color': 'orange', 'line_style': 'solid'}}


    plot('epoch', 'f1 score', plot_args)