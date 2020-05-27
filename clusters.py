"""Finds and analyzes clusters in cell images"""

import argparse
import math
import os
import sys

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial
import skimage.io
from matplotlib.patches import Ellipse
from skimage.measure import EllipseModel
from sklearn.cluster import DBSCAN, OPTICS


def parse_arguments():
    """ Parses command line to obtain the root directory.
    Returns a dictionary with the 'input_res' and
    'output_res' keys. These keys define the input and
    output directories selected by the user.
    """
    describe = "Show clusters for each image"
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required arguments")

    help_i = "CSV directory containing coordinates of images"
    required.add_argument("-csv", "--input_dir", help=help_i, type=str, required=True)

    help_o = "Output directory that will show clusters for each image"
    required.add_argument("-clusteredImages", "--clustered_images_dir",
                          help=help_o, type=str, required=True)

    help_small = "path to statistics directory for small clusters"
    required.add_argument("-stats_s", "--statistics_dir_s",
                          help=help_small, type=str, required=True)

    help_large = "path to statistics directory for larg clusters"
    required.add_argument("-stats_l", "--statistics_dir_l",
                          help=help_large, type=str, required=True)

    help_split = "this value separates the cutt-off of large \
    clusters and small clusters; default is 10"
    required.add_argument("-split", "--split", help=help_split,
                          type=int, required=False)

    help_im = "images_dir"
    required.add_argument("-images", "--images_dir",
                          help=help_im, type=str, required=True)

    help_e = "epsilon value"
    required.add_argument("-epsilon", "--epsilon",
                          help=help_e, type=int, required=False)

    help_c = "clustering method"
    required.add_argument("-method", "--method", help=help_c,
                          type=str, required=False)

    help_min = "minimum cluster size"
    required.add_argument("-min_size", "--min_size", help=help_min,
                          type=int, required=False)

    help_outline = "choices of 0-2; 0 to show ellipses, 1 to show convex hulls,\
     and 2 to show both ellipses and " \
                   "convex hulls "
    required.add_argument("-outline", "--outline", help=help_outline,
                          type=int, required=False)

    args = parser.parse_args()

    print(args.outline)
    if args.method is not None:
        if args.method.lower() not in ['dbscan', 'optics', 'hdbscan']:
            sys.exit("ERROR: inputted clustering method not available")
    if args.min_size is not None:
        if args.min_size <= 4:
            sys.exit("ERROR: min size must be greater than four")

    assert os.path.isdir(args.input_dir), "Unable to find input directory: {}". \
        format(args.input_dir)

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
    """Reads a list of coordinates from csv file and returns coordinates"""
    coordinates = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1, 2))
    coordinates = np.array(coordinates, dtype='float64')
    return coordinates


def get_clusters(data, epsilon=None, min_size=None, cluster_method='dbscan'):
    """Performs the clustering algorithm and plots clusters"""
    if epsilon is None:
        epsilon = 35
    if min_size is None:
        min_size = 6
    if cluster_method is not None:
        cluster_method = cluster_method.lower()
    if cluster_method == 'dbscan' or cluster_method is None:
        clust = DBSCAN(eps=epsilon, min_samples=min_size).fit(data)
    elif cluster_method == 'optics':
        clust = OPTICS(min_samples=min_size).fit(data)
    elif cluster_method == 'hdbscan':
        clust = hdbscan.HDBSCAN(min_cluster_size=min_size, allow_single_cluster=False).fit(data)
    else:
        sys.exit("ERROR")
    labels = clust.labels_

    core_samples_mask = np.zeros_like(clust.labels_, dtype=bool)
    core_samples_mask[np.where(clust.labels_ != -1)] = True

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    colors = plt.cm.get_cmap("Spectral")(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        x_y = data[class_member_mask & core_samples_mask]
        plt.plot(x_y[:, 0], x_y[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)

    plt.title('Estimated number of clusters: %d' % n_clusters)
    return labels


def cluster_info(data, paths,
                 dimensions, axes, optional_params):
    """Determines statistics for each cluster and outputs to small and large stats directories"""
    unique_labels = set(data["cluster_labels"][data["cluster_labels"] != -1])
    info_s = pd.DataFrame(
        columns=['label', 'centroid', 'area', 'n_nuclei', 'nuclei_density',
                 '    eccentricity', 'ellipticity'],
        dtype='int64', copy=True)
    info_l = pd.DataFrame(
        columns=['label', 'centroid', 'area', 'n_nuclei', 'nuclei_density',
                 '    eccentricity', 'ellipticity'],
        dtype='int64', copy=True)
    info_s.centroid.astype(str)
    info_l.centroid.astype(str)
    info_s.label.round(0)
    info_l.label.round(0)
    stats = {}
    if optional_params["split"] is None:
        optional_params["split"] = 10
    stats["total_area"] = dimensions["height"] * dimensions["width"]
    for label in unique_labels:  # for every cluster
        indices = np.where(data["cluster_labels"] == label)
        cluster_coords = np.array(data["coordinates"][indices], dtype='float32')
        stats["n_indices"] = len(cluster_coords)
        # centroid
        stats["x_average"] = sum(cluster_coords[:, 0]) / stats["n_indices"]
        stats["y_average"] = sum(cluster_coords[:, 1]) / stats["n_indices"]

        # if the cluster has more than two nuclei, constructs a polygon
        if stats["n_indices"] > 3:
            hull = spatial.ConvexHull(cluster_coords)
            stats = compute_hull_and_outline(hull, cluster_coords, stats, optional_params, axes)

        stats["area"] = (hull.volume / stats["total_area"]) * 100
        stats["density"] = stats["area"] / stats["total_area"]
        if stats["n_indices"] >= optional_params["split"]:
            info_l = info_l.append(
                {'centroid': f'({format(stats["x_average"], ".2f")},'
                             f'{format(stats["y_average"], ".2f")})',
                 'label': int(label),
                 'area': format(stats["area"], '.4f'),
                 'n_nuclei': format(stats["n_indices"], '.6f'),
                 'nuclei_density': format(stats["density"], '.15f'),
                 '    eccentricity': stats["eccentricity"],
                 'ellipticity': stats["ellipticity"]},
                ignore_index=True)
        else:
            info_s = info_s.append(
                {'centroid': f'({format(stats["x_average"], ".2f")},'
                             f'{format(stats["y_average"], ".2f")})',
                 'label': int(label),
                 'area': format(stats["area"], '.4f'),
                 'n_nuclei': format(stats["n_indices"], '.6f'),
                 'nuclei_density': format(stats["density"], '.15f'),
                 '    eccentricity': stats["eccentricity"],
                 'ellipticity': stats["ellipticity"]},
                ignore_index=True)

    info_l.to_csv(paths["large_cluster_stats"], index=False, sep='\t')
    info_s.to_csv(paths["small_cluster_stats"], index=False, sep='\t')


def compute_hull_and_outline(hull, cluster_coords, stats, optional_params, axes):
    """Computes convex hull, computes eccentricity, and adds outline
    Returns stats"""
    hull_points = cluster_coords[hull.vertices]
    ell = EllipseModel()
    if ell.estimate(hull_points):
        x_center, y_center, a_radius, b_radius, theta = ell.params
        stats["ellipticity"] = compute_ellipticity_elliptic_variance(
            cluster_coords, [stats["x_average"], stats["y_average"]])
        degrees = math.degrees(theta)
        if a_radius > b_radius:
            stats["eccentricity"] = math.sqrt(1 - (b_radius / a_radius) ** 2)
            ellipse = Ellipse((x_center, y_center), 2 * a_radius, 2 * b_radius, degrees, fill=False)
        elif b_radius > a_radius:
            stats["eccentricity"] = math.sqrt(1 - (a_radius / b_radius) ** 2)
            ellipse = Ellipse((x_center, y_center), 2 * a_radius, 2 * b_radius, degrees, fill=False)
        else:
            stats["eccentricity"] = 1
            ellipse = Ellipse((x_center, y_center), 2 * a_radius, 2 * b_radius, degrees, fill=False)
        if optional_params["outline"] == 2:
            for simplex in hull.simplices:
                axes.plot(cluster_coords[simplex, 0], cluster_coords[simplex, 1],
                          'k-', color='green')
            axes.add_patch(ellipse)
        elif optional_params["outline"] == 1:
            for simplex in hull.simplices:
                axes.plot(cluster_coords[simplex, 0], cluster_coords[simplex, 1], 'k-')
        elif optional_params["outline"] == 0:
            axes.add_patch(ellipse)
    else:
        stats["eccentricity"] = None
        stats["ellipticity"] = None
    return stats


def compute_ellipticity_moment_based(cluster_coords, centroid):
    """Returns an area based ellipticity measure using moments"""
    result = 0
    i = 0
    while i <= 4:
        j = 0
        while i + j <= 4:
            result += (central_moment_derivative(cluster_coords, centroid, i, j) -
                       central_moment(cluster_coords, centroid, i, j)) ** 2
            j += 1
        i += 1
    return result


def compute_ellipticity_affine_transform(cluster_coords, centroid):
    """Returns an ellipticity measure based on affine moment invariants"""
    central_moment_20 = central_moment(cluster_coords, centroid, 2, 0)
    central_moment_02 = central_moment(cluster_coords, centroid, 0, 2)
    central_moment_11 = central_moment(cluster_coords, centroid, 1, 1)
    central_moment_00 = central_moment(cluster_coords, centroid, 0, 0)
    affine_moment_invariant = (central_moment_20 * central_moment_02 - (central_moment_11 ** 2)) / \
                              (central_moment_00 ** 4)
    print(affine_moment_invariant)
    if affine_moment_invariant <= 1 / (16 * (math.pi ** 2)):
        return 16 * (math.pi ** 2) * affine_moment_invariant
    return 1 / (16 * (math.pi ** 2) * affine_moment_invariant)


def central_moment(cluster_coords, centroid, i, j):
    """Returns the central moment"""
    result = 0
    for coord in cluster_coords:
        result += ((coord[0] - centroid[0]) ** i) * ((coord[1] - centroid[1]) ** j)
    return result


def central_moment_derivative(cluster_coords, centroid, i, j):
    """Returns the derivative of the central moment"""
    result = 0
    for coord in cluster_coords:
        result += (((coord[0] - centroid[0]) ** i) * j * ((coord[1] - centroid[1]) ** (j - 1))) + \
                  (((coord[1] - centroid[1]) ** j) * i * ((coord[0] - centroid[0]) ** (i - 1)))
    return result


def compute_ellipticity_elliptic_variance(cluster_coords, centroid):
    """returns ellipticity using the method of elliptiv variance"""
    covariance = np.zeros((2, 2))
    centroid = np.asarray(centroid)
    for coord in cluster_coords:
        temp = np.subtract(coord, centroid)
        covariance = np.add(covariance, np.outer(temp, temp))
    covariance = np.divide(covariance, len(cluster_coords))

    mean_radius = 0
    for coord in cluster_coords:
        temp = np.subtract(coord, centroid)
        temp = temp.T
        mean_radius += (np.dot(np.dot(temp.T, np.linalg.inv(covariance)), temp)) ** (1/2)
    mean_radius /= len(cluster_coords)

    elliptic_variance = 0
    for coord in cluster_coords:
        temp = np.subtract(coord, centroid)
        temp = temp.T
        elliptic_variance += (((np.dot(np.dot(temp.T, np.linalg.inv(covariance)), temp))
                               ** (1/2)) - mean_radius) ** 2
        elliptic_variance /= len(cluster_coords) * (mean_radius ** 2)

    ellipticity = 1 / (1 + elliptic_variance)
    return ellipticity



def extract_clusters(user_options):
    """initiates clustering analysis of entire directories"""
    for csv_name in os.listdir(user_options["input_res"]):
        if csv_name.split(".")[-1] in ["csv"]:
            paths = {}
            paths["csv_path"] = os.path.join(user_options["input_res"], csv_name)

            paths["small_cluster_stats"] = user_options["statistics_s"] + '/' + csv_name
            paths["large_cluster_stats"] = user_options["statistics_l"] + '/' + csv_name
            img_cluster_name = csv_name.split(".")[0] + '.png'
            paths["img_cluster_path"] = user_options["clustered_images"] + '/' + img_cluster_name
            coordinates = read_csv(paths["csv_path"])
            img_path = user_options["images"] + '/' + csv_name.split(".")[0] + '.tiff'
            img = skimage.io.imread(img_path)
            dimensions = {}
            if len(img.shape) > 2:
                dimensions["height"], dimensions["width"], dimensions["channels"] = img.shape
            else:
                dimensions["height"], dimensions["width"] = img.shape
                dimensions["channels"] = 1
            axes = plt.gca()
            axes.set_ylim((dimensions["height"], 0), auto=False)
            axes.set_xlim((0, dimensions["width"]), auto=False)
            plt.imshow(img)

            labels = get_clusters(coordinates, user_options["epsilon"],
                                  cluster_method=user_options["method"])
            data = {"coordinates": coordinates,
                    "cluster_labels": labels}
            optional_params = {"outline": user_options["outline"],
                               "split": user_options["split"]}
            cluster_info(data, paths, dimensions, axes, optional_params)
            plt.savefig(paths["img_cluster_path"])
            plt.clf()


if __name__ == "__main__":
    extract_clusters(parse_arguments())
