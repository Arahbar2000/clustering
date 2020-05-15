import os
import argparse
import numpy as np
import skimage.io

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import scipy as sc
import os
import skimage.io
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull, distance
import math
from skimage import filters
from sklearn.cluster import OPTICS
import hdbscan
import skimage.measure as measure
from numpy.linalg import eig, inv
import math
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import sys


def parse_arguments():
    """ Parses command line to obtain the root directory. Returns a dictionary with the 'input_res' and
    'output_res' keys. These keys define the input and output directories selected by the user.
    """
    describe = "Show clusters for each image"
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required arguments")

    help_i = "CSV directory containing coordinates of images"
    required.add_argument("-csv", "--input_dir", help=help_i, type=str, required=True)

    help_o = "Output directory that will show clusters for each image"
    required.add_argument("-clusteredImages", "--clustered_images_dir", help=help_o, type=str, required=True)

    help_e = "epsilon value"
    required.add_argument("-epsilon", "--epsilon", help=help_e, type=int, required=True)

    help_c = "clustering method"
    required.add_argument("-method", "--method", help=help_c, type=str, required=True)

    help_os = "statistics_dir"
    required.add_argument("-stats", "--statistics_dir", help=help_os, type=str, required=True)

    help_im = "images_dir"
    required.add_argument("-images", "--images_dir", help=help_im, type=str, required=True)

    help_min = "minimum cluster size"
    required.add_argument("-min_size", "--min_size", help=help_min, type=int, required=True)

    help_outline = "choices of 0-3; 0 to not show any outline, 1 to show ellipses, 2 to show convex hulls, and 3 to show both ellipses and convex hulls"
    required.add_argument("-outline", "--outline", help=help_outline, type=int, required=True)






    args = parser.parse_args()
    if args.method.lower() not in ['dbscan', 'optics', 'hdbscan']:
    	sys.exit("ERROR: inputted clustering method not available")

    if args.min_size <= 4:
    	sys.exit("ERROR: min size must be greater than four")


    assert os.path.isdir(args.input_dir), "Unable to find input directory: {}".format(args.input_dir)

    if not os.path.isdir(args.clustered_images_dir):
        os.makedirs(args.clustered_images_dir)

    if not os.path.isdir(args.statistics_dir):
    	os.makedirs(args.statistics_dir)

    if not os.path.isdir(args.images_dir):
    	sys.exit("ERROR: images directory does not exist!")

    return {
        "input_res": args.input_dir,
        "clustered_images": args.clustered_images_dir,
        "epsilon": args.epsilon,
        "method": args.method,
        "statistics": args.statistics_dir,
        "images": args.images_dir,
        "min_size": args.min_size,
        "outline": args.outline
    }

def read_csv(csv_path):
	coordinates = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1,2))
	return coordinates

def get_clusters(data, e = 35, min_size = 6, clusterMethod='dbscan'):
	if clusterMethod == 'dbscan':
		clust = DBSCAN(eps=e, min_samples=min_size).fit(data)
	elif clusterMethod == 'optics':
		clust = OPTICS(min_samples = min_size).fit(data)
	elif clusterMethod == 'hdbscan':
		clust = hdbscan.HDBSCAN(min_cluster_size=min_size, allow_single_cluster=False).fit(data)
	else: sys.exit("ERROR")
	labels = clust.labels_

	core_samples_mask = np.zeros_like(clust.labels_, dtype=bool)
	core_samples_mask[np.where(clust.labels_ != -1)] = True


	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
	colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
		if k == -1:
			col = [0, 0, 0, 1]
		class_member_mask = (labels == k)
		xy = data[class_member_mask & core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)
		xy = data[class_member_mask & ~core_samples_mask]
		# plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

	plt.title('Estimated number of clusters: %d' % n_clusters)
	return labels

def clusterInfo(data, labels, out_path, h, w, axes, outline=0,):
	data = np.array(data, dtype='float64')
	unique_labels = set(labels[labels != -1])
	info = pd.DataFrame(columns = ['label', 'centroid', 'area','n_nuclei', 'nuclei_density', '    eccentricity'], dtype='int64', copy=True)
	info.centroid.astype(str)
	info.label.round(0)
	n_clusters_ = len(unique_labels)
	e = None
	total_area = h*w
	for label in unique_labels: # for every cluster
		indices =  np.where(labels == label)
		clusterIndices = np.array(data[indices], dtype='float32')
		sum = 0
		# centroid
		x_average = 0
		y_average = 0

		# x range and y range of coordinates in cluster
		x = clusterIndices[:, 0]
		y = clusterIndices[:, 1]
		n_indices = 0
		area = 0
		density = 0

		# computes centroid
		for coord in clusterIndices:
			x_average += coord[0]
			y_average += coord[1]
			n_indices += 1
		x_average /= n_indices
		y_average /= n_indices

		# if the cluster has more than two nuclei, constructs a polygon
		if n_indices > 3:
			hull = ConvexHull(clusterIndices)
			borderIndices = hull.vertices
			hullPoints = clusterIndices[borderIndices]
			xh = hullPoints[:, 0]
			yh = hullPoints[:, 1]
			ell = EllipseModel()
			ellipse = None
			if ell.estimate(hullPoints):
				xc, yc, a, b, theta = ell.params
				degrees = math.degrees(theta)
				if a > b:
					e = math.sqrt(1 - (b/a)**2)
					ellipse = Ellipse((xc,yc), 2*a, 2*b, degrees, fill=False)
				elif b > a:
					e = math.sqrt(1 - (a/b)**2)
					ellipse = Ellipse((xc,yc), 2*a, 2*b, degrees, fill=False)
				else:
					e = 1
					ellipse = Ellipse((xc,yc), 2*a, 2*b, degrees, fill=False)
				if outline == 3:
					for simplex in hull.simplices:
						axes.plot(clusterIndices[simplex, 0], clusterIndices[simplex, 1], 'k-', color='green')
					axes.add_patch(ellipse)
				elif outline == 2:
					for simplex in hull.simplices:
						axes.plot(clusterIndices[simplex, 0], clusterIndices[simplex, 1], 'k-')
				elif outline == 1:
					axes.add_patch(ellipse)

					

		area = (hull.volume / total_area)*100 
		density = area / total_area
		info = info.append({'label': int(label), 'centroid': '({},{})'.format(format(x_average, '.2f'), format(y_average, '.2f')), 'area': format(area, '.4f'), 'n_nuclei': format(n_indices, '.6f'), 'nuclei_density': format(density, '.15f'), '    eccentricity': e}, ignore_index=True)
	info.to_csv(out_path, index=None, sep='\t')

def extract_clusters(user_options):
	for csv_name in os.listdir(user_options["input_res"]):
		if csv_name.split(".")[-1] in ["csv"]:
			csv_path = os.path.join(user_options["input_res"], csv_name)
			stats_path = user_options["statistics"] + '/' + csv_name
			img_cluster_name = csv_name.split(".")[0] + '.png'
			img_cluster_path = user_options["clustered_images"] + '/' + img_cluster_name
			coordinates = read_csv(csv_path)
			img_name = csv_name.split(".")[0]
			img_path = user_options["images"] + '/' + img_name + '.tiff'
			img = skimage.io.imread(img_path)
			if len(img.shape) > 2:
   				h, w, c = img.shape
			else:
				h, w = img.shape
				c = 1
			axes = plt.gca()
			axes.set_ylim((h, 0), auto=False)
			axes.set_xlim((0, w), auto=False)
			plt.imshow(img)

			labels = get_clusters(coordinates, user_options["epsilon"], clusterMethod=user_options["method"].lower())
			clusterInfo(coordinates, labels, stats_path, h ,w, axes, outline=user_options["outline"])
			plt.savefig(img_cluster_path)
			plt.clf()

if __name__ == "__main__":
    extract_clusters(parse_arguments())