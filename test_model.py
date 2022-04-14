import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import DBSCAN, MeanShift
import shutil
import os

from dataset import DataSet
from lanenet import LaneNet
from process_data import process_dataset
from utils import get_config
from compute_loss import binary_segmentation_loss, instance_segmentation_loss, accuracy

TEST_DATASET = "test"
EX_TEST_DATASET = "ex_test"


###############################################################################
# Testing Script

def test_model(dataset, config, model_name):
    path_root =  config.path.root[dataset]

    # Create the dataset if it doesn't exist
    if (os.path.isdir(path_root) and path_root+dataset+"_data.txt"
        and os.path.isdir(path_root+config.path.image.src)
        and os.path.isdir(path_root+config.path.image.binary)
        and os.path.isdir(path_root+config.path.image.instance)): 
        print("Dataset already exists. Skipping dataset creation.")
    else:
        print("Creating dataset.")
        process_dataset(dataset)

    # Create the test output directory
    path_results = config.path.test_results
    if os.path.isdir(path_results): shutil.rmtree(path_results)
    os.makedirs(os.path.dirname(path_results), exist_ok=True)
    os.makedirs(os.path.dirname(path_results+config.path.image.src), exist_ok=True)
    os.makedirs(os.path.dirname(path_results+config.path.image.results), exist_ok=True)
    
    # Load the dataset
    test_dataset = DataSet(dataset, config)
    test_dataset.load_dataset()

    # Create the model
    model = tf.keras.models.load_model(config.path.models+model_name,
        custom_objects={'binary_segmentation_loss': binary_segmentation_loss,
        'instance_segmentation_loss': instance_segmentation_loss, 'accuracy': accuracy})
    model.summary()

    # Test the model
    batch_size = config.params.test.batch_size
    test_epochs = config.params.test.epochs
    file_ctr = 0

    for epoch in range(test_epochs):
        src_imgs, binary_imgs, instance_imgs = test_dataset.next_batch(batch_size)
        
        binary_out, instance_out = model.predict(np.array(src_imgs))
        binary_out = np.array([np.argmax(x, axis=-1) for x in binary_out])

        for idx, binary_img in enumerate(binary_out):
            binary_img = postProcess(np.uint8(binary_imgs[idx]))
            lane_mask = create_lane_mask(binary_img, instance_out[idx])
            
            src_img = src_imgs[idx]
            
            output = cv.addWeighted(src_img, 1.0, lane_mask, 1.0, 0)

            cv.imwrite(path_results+config.path.image.src+"{}.jpg".format(file_ctr), src_img)
            cv.imwrite(path_results+config.path.image.results+"{}.jpg".format(file_ctr), output)
            file_ctr += 1

    return


###############################################################################
# Post-processing and clustering

def postProcess(image):
    threshold = 15

    # Morphology
    morphology = cv.morphologyEx(image, cv.MORPH_CLOSE, 
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Connected component analysis
    mor_gray = cv.cvtColor(morphology, cv.COLOR_BGR2GRAY) if len(morphology.shape) == 3 else morphology
    components = cv.connectedComponentsWithStats(mor_gray, 8, ltype=cv.CV_32S)

    # Sort connected domains and delete too small connected domains
    for index, stat in enumerate(components[2]):
        if stat[4] <= threshold:
            idx = np.where(components[1] == index)
            morphology[idx] = 0

    return morphology


def create_lane_mask(binary_img, instance_img):
    # Colors for lane instances
    lane_colors = [(0,0,204), (0,204,204), (0,204,0), (204,204,0), (204,0,0), (204,0,204)]
    
    lane_pixels = np.where(binary_img == 1)

    lane_features = []
    lane_coordinates = []
    for i in range(len(lane_pixels[0])):
        lane_features.append(instance_img[lane_pixels[0][i], lane_pixels[1][i]])
        lane_coordinates.append([lane_pixels[0][i], lane_pixels[1][i]])

    lane_features = np.array(lane_features, np.float32)
    lane_coordinates = np.array(lane_coordinates, np.int64)

    # Cluster the lane instances
    # db = DBSCAN(eps=0.7, min_samples=200).fit(lane_features)
    # labels = db.labels_
    # cluster_centers = db.components_
    # num_clusters = len([tmp for tmp in np.unique(labels) if tmp != -1])

    meanShift = MeanShift(bandwidth=1.5, bin_seeding=True)
    err = False
    try:
        meanShift.fit(lane_features)
    except ValueError as e:
        err = True
    labels = meanShift.labels_ if not err else []
    cluster_centers = meanShift.cluster_centers_ if not err else []
    num_clusters = cluster_centers.shape[0] if not err else 0

    # Choose 5 clusters with the highest cluster centers
    if num_clusters > 5:
        cluster_sample_nums = []
        for i in range(num_clusters):
            cluster_sample_nums.append(len(np.where(labels == i)[0]))
        sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
        cluster_index = np.array(range(num_clusters))[sort_idx[0:5]]
    else:
        cluster_index = range(num_clusters)

    # Create the lane mask
    lane_mask = np.zeros(shape=[binary_img.shape[0], binary_img.shape[1], 3], dtype=np.uint8)
    
    for idx, i in enumerate(cluster_index):
            pixels = np.where(labels == i)
            coord = lane_coordinates[pixels]
            coord = np.flip(coord, axis=1)
            cv.polylines(lane_mask, np.array([coord]), False, lane_colors[idx], 2)

    return lane_mask


if __name__ == "__main__":
    config = get_config()
    test_model(EX_TEST_DATASET, config, model_name="model#2")