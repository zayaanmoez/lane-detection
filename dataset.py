import numpy as np
import cv2 as cv
import os

class DataSet():
    def __init__(self, dataset, config):
        self.dataset_name = dataset
        self.path_config = config.path
        self.path_root = config.path.root[dataset]
        self.input_width = config.params.input_width
        self.input_height = config.params.input_height


    def load_dataset(self):
        data_file = self.path_root+self.dataset_name+"_data.txt"
        assert os.path.isdir(self.path_root) and os.path.isfile(data_file), "Dataset not found."

        with open(self.path_root+self.dataset_name+"_data.txt", 'r') as f:
            f.readline()
            self.num_files = int(f.readline().split()[1])
            f.close()

        print("Dataset loaded. Number of files:", self.num_files)


    def next_batch(self, batch_size):
        src_imgs = []
        binary_imgs = []
        instance_imgs = []

        path_src =self.path_root+self.path_config.image.src
        path_binary = self.path_root+self.path_config.image.binary
        path_instance = self.path_root+self.path_config.image.instance
        ext = self.path_config.image.ext

        random_sample = np.random.permutation(self.num_files)
        
        files, i = 0, 0
        while files < batch_size:
            try:
                src_img = cv.imread(path_src+str(random_sample[i])+ext, cv.IMREAD_COLOR)
                binary_img = cv.imread(path_binary+str(random_sample[i])+ext, cv.IMREAD_GRAYSCALE)
                instance_img = cv.imread(path_instance+str(random_sample[i])+ext, cv.IMREAD_GRAYSCALE)
            except:
                i += 1
                continue

            src_imgs.append(src_img)
            binary_imgs.append(binary_img)
            instance_imgs.append(instance_img)
            files += 1 
            i += 1

        # Resize the batch images
        src_imgs = [cv.resize(src_img, (self.input_width, self.input_height), 
            interpolation=cv.INTER_AREA)
            for src_img in src_imgs]

        binary_imgs = [cv.resize(binary_img, (self.input_width, self.input_height),
            interpolation=cv.INTER_AREA)
            for binary_img in binary_imgs]

        instance_imgs = [cv.resize(instance_img, (self.input_width, self.input_height),
            interpolation=cv.INTER_AREA)
            for instance_img in instance_imgs]

        return src_imgs, binary_imgs, instance_imgs