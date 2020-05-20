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
from math import cos, sin


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

    help_small = "path to statistics directory for small clusters"
    required.add_argument("-stats_s", "--statistics_dir_s", help=help_small, type=str, required=True)

    help_large = "path to statistics directory for larg clusters"
    required.add_argument("-stats_l", "--statistics_dir_l", help=help_large, type=str, required=True)

    help_split = "this value separates the cutt-off of large clusters and small clusters; default is 10"
    required.add_argument("-split", "--split", help=help_split, type=int, required=False)

    help_im = "images_dir"
    required.add_argument("-images", "--images_dir", help=help_im, type=str, required=True)

    help_e = "epsilon value"
    required.add_argument("-epsilon", "--epsilon", help=help_e, type=int, required=False)

    help_c = "clustering method"
    required.add_argument("-method", "--method", help=help_c, type=str, required=False)

    help_min = "minimum cluster size"
    required.add_argument("-min_size", "--min_size", help=help_min, type=int, required=False)

    help_outline = "choices of 0-2; 0 to show ellipses, 1 to show convex hulls, and 2 to show both ellipses and convex hulls"
    required.add_argument("-outline", "--outline", help=help_outline, type=int, required=False)






    args = parser.parse_args()

    print(args.outline)
    if args.method != None:
    	if args.method.lower() not in ['dbscan', 'optics', 'hdbscan']:
    		sys.exit("ERROR: inputted clustering method not available")
    if args.min_size != None:
    	if args.min_size <= 4:
    		sys.exit("ERROR: min size must be greater than four")


    assert os.path.isdir(args.input_dir), "Unable to find input directory: {}".format(args.input_dir)

    if not os.path.isdir(args.clustered_images_dir):
        os.makedirs(args.clustered_images_dir)

    if not os.path.isdir(args.statistics_dir_s):
    	os.makedirs(args.statistics_dir_s)
    if not os.path.isdir(args.statistics_dir_l):
    	os.makedirs(args.statistics_dir_l)
    if not os.path.isdir(args.images_dir):
    	sys.exit("ERROR: images directory does not exist!")

    return {
        "input_res": args.input_dir,
        "clustered_images": args.clustered_images_dir,
        "epsilon": args.epsilon,
        "method": args.method,
        "statistics_s": args.statistics_dir_s,
        "statistics_l": args.statistics_dir_l,
        "images": args.images_dir,
        "min_size": args.min_size,
        "outline": args.outline,
        "split": args.split
    }

def read_csv(csv_path):
	coordinates = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1,2))
	return coordinates

def get_clusters(data, e = None, min_size = None, clusterMethod='dbscan'):
	if e == None:
		e = 35
	if min_size == None:
		min_size = 6
	if clusterMethod != None:
		clusterMethod = clusterMethod.lower()
	if clusterMethod == 'dbscan' or clusterMethod == None:
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

def clusterInfo(data, labels, stats_path_s, stats_path_l, h, w, axes, outline=None, split=None):
	data = np.array(data, dtype='float64')
	unique_labels = set(labels[labels != -1])
	info_s = pd.DataFrame(columns = ['label', 'centroid', 'area','n_nuclei', 'nuclei_density', '    eccentricity', 'ellipticity'], dtype='int64', copy=True)
	info_l = pd.DataFrame(columns = ['label', 'centroid', 'area','n_nuclei', 'nuclei_density', '    eccentricity', 'ellipticity'], dtype='int64', copy=True)
	info_s.centroid.astype(str)
	info_l.centroid.astype(str)
	info_s.label.round(0)
	info_l.label.round(0)
	n_clusters_ = len(unique_labels)
	e = None
	ellipticity = None
	if split == None:
		split = 10
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
				ellipticity = getEllipticity(hullPoints, xc, yc, a, b, theta)
				# print(ellipticity)
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
				if outline == 2:
					for simplex in hull.simplices:
						axes.plot(clusterIndices[simplex, 0], clusterIndices[simplex, 1], 'k-', color='green')
					axes.add_patch(ellipse)
				elif outline == 1:
					for simplex in hull.simplices:
						axes.plot(clusterIndices[simplex, 0], clusterIndices[simplex, 1], 'k-')
				elif outline == 0:
					axes.add_patch(ellipse)
				

					

		area = (hull.volume / total_area)*100 
		density = area / total_area
		if(n_indices >= split):
			info_l = info_l.append({'label': int(label), 'centroid': '({},{})'.format(format(x_average, '.2f'), format(y_average, '.2f')), 'area': format(area, '.4f'), 'n_nuclei': format(n_indices, '.6f'), 'nuclei_density': format(density, '.15f'), '    eccentricity': e, 'ellipticity': ellipticity}, ignore_index=True)
		else:
			info_s = info_s.append({'label': int(label), 'centroid': '({},{})'.format(format(x_average, '.2f'), format(y_average, '.2f')), 'area': format(area, '.4f'), 'n_nuclei': format(n_indices, '.6f'), 'nuclei_density': format(density, '.15f'), '    eccentricity': e, 'ellipticity': ellipticity}, ignore_index=True)
	info_l.to_csv(stats_path_l, index=None, sep='\t')
	info_s.to_csv(stats_path_s, index=None, sep='\t')

def extract_clusters(user_options):
	for csv_name in os.listdir(user_options["input_res"]):
		if csv_name.split(".")[-1] in ["csv"]:
			csv_path = os.path.join(user_options["input_res"], csv_name)
			stats_path_s = user_options["statistics_s"] + '/' + csv_name
			stats_path_l = user_options["statistics_l"] + '/' + csv_name
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

			labels = get_clusters(coordinates, user_options["epsilon"], clusterMethod=user_options["method"])
			clusterInfo(coordinates, labels, stats_path_s, stats_path_l, h ,w, axes, outline=user_options["outline"], split=user_options["split"])
			plt.savefig(img_cluster_path)
			plt.clf()

# computes the average distance ratio to ellipse and the center, which provides a measure of ellipticity
# method obtained from milos stojmenovic in "Direct Eclipse Fitting and Measuring Based on Shape Boundaries"
def getEllipticity(coords, xc, yc, a, b, theta):
	sum = 0
	for point in coords:
		intersect = getIntersection(xc, yc, a, b, theta, point)
		if intersect == None:
			return None
		distIntersect = dist([point[0], point[1]], [intersect[0], intersect[1]])
		distCenter = dist([point[0], point[1]], [xc, yc])
		sum += (distIntersect/distCenter)
	return sum / len(coords)

def getIntersection(xc, yc, a, b, theta, point):
	m = (point[1] - yc) / (point[0] - xc) # slope
	bi = point[1] - (m*point[0]) # y-intercept
	if a < b:
		t = b
		b = a
		a = t
	# coefficients of quadratic equation
	A = (((cos(theta)**2))+((2*m*cos(theta)*sin(theta))) + ((m**2)*sin(theta)))/(a**2) + (((sin(theta)**2))+((2*m*cos(theta)*sin(theta))) + ((m**2)*cos(theta)))/(b**2)
	B = (((-2*xc*(cos(theta)**2))+(2*bi*cos(theta)*sin(theta))-(2*yc*cos(theta)*sin(theta))-(2*m*xc*cos(theta)*sin(theta))+(2*m*bi*(sin(theta**2)))-(2*m*yc*(sin(theta)**2)))/(a**2)) + (((-2*xc*(sin(theta)**2))+(2*bi*cos(theta)*sin(theta))-(2*yc*cos(theta)*sin(theta))-(2*m*xc*cos(theta)*sin(theta))+(2*m*bi*(cos(theta**2)))-(2*m*yc*(cos(theta)**2)))/(b**2))
	C = ((((xc**2)*(cos(theta)**2))-(2*bi*xc*cos(theta)*sin(theta))+(2*xc*yc*cos(theta)*sin(theta))+(bi**2)*(sin(theta)**2)-(2*bi*yc*(sin(theta)**2))+((yc**2)*(sin(theta)**2)))/(a**2)) + ((((xc**2)*(sin(theta)**2))-(2*bi*xc*cos(theta)*sin(theta))+(2*xc*yc*cos(theta)*sin(theta))+(bi**2)*(cos(theta)**2)-(2*bi*yc*(cos(theta)**2))+((yc**2)*(cos(theta)**2)))/(b**2))

	print("A: {}, B: {}, C: {}, m: {}, bi: {}".format(A, B, C, m, bi))


	det = (B**2) - (4*A*C)
	intersection = 0
	if det >= 0:
		# solving for x
		xUpper = ((B*-1) + (det**(1/2))) / (2*A)
		xLower = ((B*-1) - (det**(1/2))) / (2*A)

		# using x coordinates to find y coordinates
		yUpper = m*xUpper + bi
		yLower = m*xLower + bi
		distUpper = dist([point[0], point[1]], [xUpper, yUpper])
		distLower = dist([point[0], point[1]], [xLower, yLower])

		# returning the closest intersection
		if distUpper <= distLower:
			return (xUpper, yUpper)
		else:
			return (xLower, yLower)
	else:
		return None

def dist(a, b):
	x1 = a[0]
	x2 = b[0]
	y1 = a[1]
	y2 = b[1]
	return (((x2-x1)**2) + ((y2-y1)**2)) ** (1/2)



if __name__ == "__main__":
    extract_clusters(parse_arguments())