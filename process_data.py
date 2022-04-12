import os
import json
import numpy as np
import cv2 as cv
from utils import get_config


# Process the dataset for training and testing
#   data_type: "train", "test", "ex_train", "ex_test"
#   train and test use the TuSimple dataset
#   ex_train and ex_test are an exmaple subset of the dataset

train_labels = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]
test_labels = ["test_label.json"]
example_labels = ["example_label.json"]

json_labels = {"train": train_labels, "test": test_labels, "ex_train": example_labels, "ex_test": example_labels}

path_cfg = get_config().path

def process_dataset(data_type="ex_test"):
    path_root = path_cfg.root[data_type]
    path_dataset = path_cfg.dataset[data_type]
    img_ext = path_cfg.image.ext

    # Colors for lane instance embedding
    lane_colors = [20, 70, 120, 170, 220]

    # Create the data directories
    os.makedirs(os.path.dirname(path_root), exist_ok=True)
    os.makedirs(os.path.dirname(path_root+path_cfg.image.src), exist_ok=True)
    os.makedirs(os.path.dirname(path_root+path_cfg.image.binary), exist_ok=True)
    os.makedirs(os.path.dirname(path_root+path_cfg.image.instance), exist_ok=True)

    imgCtr = 0
    # Load the dataset
    for labels_json in json_labels[data_type]:
        labels = [json.loads(line) for line in open(path_dataset + labels_json)]

        # Create the train/test data 
        for label in labels:
            try:
                lanes = label["lanes"]
                h_samples = label["h_samples"]
                raw_file = label["raw_file"]

                # Get the lane points from h_samples and lanes
                lane_list = []
                for lane in lanes:
                    lane_points = []
                    for (x, y) in zip(lane, h_samples):
                        if x >= 0: lane_points.append([x, y])
                    lane_list.append(np.array(lane_points, dtype=np.int32))

                # Load the image
                img = cv.imread(path_dataset + raw_file)
                cv.imwrite(path_root + path_cfg.image.src + str(imgCtr) + img_ext, img)

                # Polyline Annotations
                binary = np.zeros(img.shape, dtype=np.uint8)
                binary = cv.cvtColor(binary, cv.COLOR_BGR2GRAY)
                cv.polylines(binary, np.array(lane_list, dtype=object), False, (255,255,255), 5)
                cv.imwrite(path_root + path_cfg.image.binary + str(imgCtr) + img_ext, binary)

                instance = np.zeros(img.shape, dtype=np.uint8)
                instance = cv.cvtColor(instance, cv.COLOR_BGR2GRAY)
                for i in range(len(lane_list)):
                    lane_color = (lane_colors[i], lane_colors[i], lane_colors[i])
                    cv.polylines(instance, np.array([lane_list[i]]), False, lane_color, 5)
                cv.imwrite(path_root + path_cfg.image.instance + str(imgCtr) + img_ext, instance)

            except:
                print("Error json label: " + str(imgCtr))
                
            imgCtr += 1
            print("Images Processed: " + str(imgCtr), end="\r")

    with open(path_root + data_type + "_data.txt", "w") as f:
        f.write("Dataset: " + data_type + "\n")
        f.write("Images: " + str(imgCtr) + "\n")

    print("Images Processed: " + str(imgCtr))
