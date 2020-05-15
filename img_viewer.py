""" A tool for viewing images in a browser using Plotly"""


import os
import sys
import argparse
import skimage.io
import skimage.morphology
import plotly.express as px
import matplotlib.pyplot as py
import numpy as np
import pandas as pd 
import csv


IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "tif", "tiff", "JPG", "JPEG", "PNG", "TIF", "TIFF"]

def exit_wiht_error(message):
    print(message)
    sys.exit(1)



filename = ""



def parse_arguments():
    global filename
    """ Parses command line to obtain the path to an image"""
    describe = "View tiff, jpg, or png images in a browser"
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group('required arguments')

    help_i = "Path to input image"
    help_o = "path to output csv file"
    required.add_argument("-i", "--input_path", help=help_i, type=str, required=True)
    required.add_argument("-o", "--output_dir", help=help_o, type=str, required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.input_path):
        msg = "Unable to find file {}".format(args.input_path)
        exit_wiht_error(msg)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    filename = args.input_path.split("/")[1]
    filename = filename.split(".")[0] + '.csv'
    filename = args.output_dir + '/' + filename
    return args.input_path


def visualize_image(image_path):
    """ Show an image using the browser's image viewer, which is accessed via Plotly"""
    _, image_name = os.path.split(image_path)

    img_ext = image_name.split(".")[-1]
    if img_ext not in IMAGE_EXTENSIONS:
        handlers.exit_with_error("Unrecognized image format: {}".format(img_ext))


    img = skimage.io.imread(image_path)

    if len(img.shape) > 2:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1

    print("Dimensions of the loaded image: {}".format(img.shape))

    # Define title text for the chart
    size_txt = "size: {}x{}  channels: {}".format(h, w, c)
    title_txt = image_name + "  " + size_txt

    # Plot figure
    fig = px.imshow(img)
    fig.update_layout(
        title={
            'text': title_txt,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },

        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#454545"
        )
    )
    fig.show()

# global variables
fig = py.figure()
h = 0
w = 0
c = 0
counter = 0
coordinates = pd.DataFrame(columns = ['index', 'x', 'y'], dtype='int64', copy=True)

def matplot_image(image_path):
    global h,w, c
    """ Show an image using the browser's image viewer, which is accessed via Plotly"""
    _, image_name = os.path.split(image_path)

    img_ext = image_name.split(".")[-1]
    if img_ext not in IMAGE_EXTENSIONS:
        handlers.exit_with_error("Unrecognized image format: {}".format(img_ext))


    img = skimage.io.imread(image_path)

    if len(img.shape) > 2:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1

    print("Dimensions of the loaded image: {}".format(img.shape))

    # Define title text for the chart
    size_txt = "size: {}x{}  channels: {}".format(h, w, c)
    title_txt = image_name + "  " + size_txt

    # Plot figure

    fig = py.imshow(img, origin='upper')
    # py.xlim(1000, 2500)
    # py.ylim(1250, 1750)
    py.show()

def onclick(event):
    global counter, coordinates

    # appends to dataframe
    coordinates = coordinates.append({'index': counter, 'x': int(event.xdata), 'y': int(event.ydata)}, ignore_index=True)
    counter+=1

    # creates a marker
    py.plot(event.xdata, event.ydata, 'bo', markersize=5)

    # redraws the plot
    fig.canvas.draw_idle()


def onkey(event):
    global counter, coordinates
    if event.key == 'enter':

        # converts dataframe to numpy array for easier processing
        data = coordinates.to_numpy()

        # creates csv file with given coordinates
        np.savetxt(filename, data, delimiter=',', fmt='%d')
        header = [w, h]

        # adds the beginning line
        with open(filename, "r") as infile:
            reader = list(csv.reader(infile))
            reader.insert(0, header)

        with open(filename, "w") as outfile:
            writer = csv.writer(outfile)
            for line in reader:
                writer.writerow(line)
    elif event.key == 'a':
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print(event.key)




# connects event handlers to plot
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
kid = fig.canvas.mpl_connect('key_press_event', onkey)




if __name__ == "__main__":
    
    # visualize_image(parse_arguments())
    matplot_image(parse_arguments())


