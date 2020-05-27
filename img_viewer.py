""" A tool for viewing images and creating coordinates"""

import os
import sys
import argparse
import csv

import skimage.io
import skimage.morphology
import matplotlib.pyplot as py
import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "tif", "tiff", "JPG", "JPEG", "PNG", "TIF", "TIFF"]


class ImageViewer:
    """Displays an image and facilitates the process of creating coordinates"""

    def __init__(self, fig):
        self.coordinates = pd.DataFrame(columns=['index', 'x', 'y'], dtype='int64', copy=True)
        self.user_options = parse_arguments()
        self.dimensions = self.get_dimensions()
        self.counter = 0
        self.matplot_image()
        self.fig = fig
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.kid = 0

    def get_dimensions(self):
        """Returns the dimensions of an image"""
        image_name = os.path.split(self.user_options["input_path"])[1]
        img_ext = image_name.split(".")[-1]
        if img_ext not in IMAGE_EXTENSIONS:
            sys.exit("Unrecognized image format: {}".format(img_ext))

        img = skimage.io.imread(self.user_options["input_path"])

        if len(img.shape) > 2:
            height, width, channels = img.shape
        else:
            height, width = img.shape
            channels = 1

        return {"height": height,
                "width": width,
                "channels": channels}

    def matplot_image(self):
        """Displays the image"""
        image_path = self.user_options["input_path"]
        image_name = os.path.split(image_path)[1]
        img_ext = image_name.split(".")[-1]
        if img_ext not in IMAGE_EXTENSIONS:
            sys.exit("Unrecognized image format: {}".format(img_ext))
        img = skimage.io.imread(self.user_options["input_path"])
        # Plot figure
        self.fig = py.imshow(img, origin='lower')

    def onclick(self, event):
        """Creates a new coordinate based on click location and plots coordinate"""
        # appends to dataframe
        self.coordinates = self.coordinates.append(
            {'index': self.counter, 'x': int(event.xdata),
             'y': int(event.ydata)}, ignore_index=True)
        self.counter += 1

        # creates a marker
        py.plot(event.xdata, event.ydata, 'bo', markersize=5)

        # redraws the plot
        self.fig.canvas.draw_idle()

    def onkey(self, event):
        """If 'enter' is pressed, creates new csv file of coordinates.
        If 'a' is pressed, allows you to start creating new coordinates."""
        if event.key == 'enter':
            # converts dataframe to numpy array for easier processing
            data = self.coordinates.to_numpy()

            # creates csv file with given coordinates
            np.savetxt(self.user_options["output_path"], data, delimiter=',', fmt='%d')
            header = [self.dimensions["width"], self.dimensions["height"]]

            # adds the beginning line
            with open(self.user_options["output_path"], "r") as infile:
                reader = list(csv.reader(infile))
                reader.insert(0, header)

            with open(self.user_options["output_path"], "w") as outfile:
                writer = csv.writer(outfile)
                for line in reader:
                    writer.writerow(line)
        elif event.key == 'a':
            self.kid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        print(event.key)


def parse_arguments():
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
        sys.exit(msg)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    image_name = args.input_path.split("/")[1]
    output_path = image_name.split(".")[0] + '.csv'
    output_path = args.output_dir + '/' + output_path
    return {"input_path": args.input_path,
            "output_path": output_path}


if __name__ == "__main__":
    figure = py.figure()
    a = ImageViewer(figure)
    py.show()
