import matplotlib.pyplot as plt
import json
import os
import numpy as np
import cv2
import Image

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


def connected_components(path):
    img = cv2.imread(path, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    ret, labels = cv2.connectedComponents(img)

    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()

def save_image(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)


def borders(path, type=-1, add_borders=False, dtype=np.float32, swap_channel=False, border_size=92+6, darken=0.4):
    img = cv2.imread(path, type)
    if type == -1 and  not swap_channel:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if add_borders:
        shape=list(img.shape)
        shape[0] += 2*border_size
        shape[1] += 2*border_size
        border_img = np.zeros(shape)
        border_img[border_size:-border_size, border_size:-border_size, ...] = img.copy()

        left = np.fliplr(img[:, :border_size].copy())
        right = np.fliplr(img[:, -border_size:].copy())
        border_img[border_size:-border_size, :border_size]=left
        border_img[border_size:-border_size, -border_size:]=right

        up = np.flipud(border_img[border_size:2*border_size, :].copy())*darken
        down = np.flipud(border_img[-2*border_size:-border_size, :].copy())*darken

        border_img[0:border_size, :] = up
        border_img[-border_size:, :] = down

        border_img[border_size:-border_size, :border_size]= border_img[border_size:-border_size, 0:border_size]*darken
        border_img[border_size:-border_size, -border_size:]= border_img[border_size:-border_size, -border_size:]*darken
        img=border_img
        img.astype(dtype=np.uint8)

        # img_p = Image.fromarray(img, 'RGB')
        # img_p.save('my.png')
        save_image(img, 'test')


if __name__== '__main__':
    # plot_args = {'original network training': {'path': 'ugan16_max_pool/run_train-tag-f1.json', 'color': 'b', 'line_style': '--'},
    #             'original network validation': {'path': 'ugan16_max_pool/run_eval-tag-f1.json', 'color': 'b', 'line_style':'solid'},
    #             'no sparse gradient network training': {'path': 'ugan16_avg_pool/run_train-tag-f1.json', 'color': 'orange', 'line_style': '--'},
    #             'no sparse gradient network validation': {'path': 'ugan16_avg_pool/run_eval-tag-f1.json', 'color': 'orange', 'line_style': 'solid'}}

    plot_args = {'classical training': {'path': 'unet16/run_train-tag-f1.json', 'color': 'b', 'line_style': '--'},
            'classical validation': {'path': 'unet16/run_eval-tag-f1.json', 'color': 'b', 'line_style':'solid'},
            'adversarial training': {'path': 'ugan16_max_pool/run_train-tag-f1.json', 'color': 'orange', 'line_style': '--'},
            'adversarial validation': {'path': 'ugan16_max_pool/run_eval-tag-f1.json', 'color': 'orange', 'line_style': 'solid'}}

    # plot_args = {'unweighted loss training': {'path': 'unet16/run_train-tag-f1.json', 'color': 'b', 'line_style': '--'},
    #     'unweighted loss validation': {'path': 'unet16/run_eval-tag-f1.json', 'color': 'b', 'line_style':'solid'},
    #     'weighted loss training': {'path': 'unet16_weight_map/run_train-tag-f1.json', 'color': 'orange', 'line_style': '--'},
    #     'weighted loss validation': {'path': 'unet16_weight_map/run_eval-tag-f1.json', 'color': 'orange', 'line_style': 'solid'}}


    #plot('epoch', 'F1 score', plot_args)
    borders('img.tif', add_borders=True)
    #connected_components('eval_hard_2.jpg')