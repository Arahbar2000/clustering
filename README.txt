# clustering

CLUSTERING IMAGES
This program allows you to generate clusters for images and also lets you use a wide variety of parameters and options for generating clusters. To run this program type:

python3 clusters.py <options>

here is the list of required options:

-csv : path to directory containing list of csv coordinate files
-clusteredImages : path to directory to store generated images showing clusters
-images : path to directory containing stored image files
-stats_s : path to directory to store generated csv files of stats for small clusters in each image
-stats_l : path to directory to store generated csv files of stats for large clusters in each image
-shapes : path to shapes directory to store white masks of each cluster as well as the ellipses

here is the list of optional options:

-method : clustering method to be used; the methods are "dbscan", "optics", and "hdbscan"; 				default is 'dbscan'
-epsilon : int value signifying the distance threshold for clusters; default is 35
-min_size : minimum size to classify a cluster; size must be greater than four; default is 6
-split : anything greater than or equal to designates a large cluster; the default is 10
-outline : choices of 0-2:
			0: only ellipses are shown
			1: only convex hulls are shown
			2: both ellipses and convex hulls are shown

Example command:

python3 clusters.py -csv csvHighRes -clusteredImages clusteredImagesHighRes -images highResolutionImages -stats_s clusterStats_s -stats_l clusterStats_l -method dbscan -epsilon 35 -min_size 6 -outline 1 -split 15 -shapes shapes

***NOTE***: csv file names and image file names must match. The only difference is the type 			specifier ending ie. ".tiff"


ANNOTATING IMAGES

To convert an image of nuclei to a CSV file, cd into this directory and input this command:

	python3 img_viewer.py -i <path to image> -o <path to output directory to store csv file>

When the window pops up, zoom into the image, then press "a" on your keyboard so that the program knows that whenever you click on the image, you are adding a coordinate.

Click on every nuclei to add a coordinate and then press "enter" to convert image to a csv file of coordinates
