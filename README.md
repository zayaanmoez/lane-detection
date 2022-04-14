# Lane Detection

An end-to-end implementation of CNN based lane-detection with instance segementation
### Prerequisites

Python 3.5+

### Installation

1. Create virtualenv
   ```sh
   pip install virtualenv
   ```

   ```sh
   virtualenv venv --no-site-packages
   ```
   or if virtualenv --version >= 17.0
    ```sh
   virtualenv venv
   ```

2. Start venv
   Linux/Unix:
   ```sh
   source venv/bin/activate
   ```
   Powershell:
   ```sh
   venv/Scripts/activate.ps1
   ```

2. Install PIP packages
   ```sh
   pip install -r requirements.txt
   ```

### Usage

-> Run the train_model.py file
   ```sh
   python train_model.py
   ```

-> Run the test_model.py file. The results are generated in test_results/results directory.
   ```sh
   python test_model.py
   ```

- The processd dataset is generated automatically by the train and test functions.
- In the main of train_model file, specify TRAIN_DATASET for the tusimple train dataset or EX_TRAIN_DATASET
    for the example dataset.
- In the main of test_model file, specify TEST_DATASET for the tusimple test dataset or EX_TEST_DATASET
    for the exmaple dataset.
- Tusimple dataset: https://github.com/TuSimple/tusimple-benchmark/issues/3

### References

- Neven, Davy & Brabandere, Bert & Georgoulis, Stamatios & Proesmans, Marc & Van Gool, Luc. (2018). Towards End-to-End Lane Detection: An Instance Segmentation Approach. 286-291. 10.1109/IVS.2018.8500547.

- E. Shelhamer, J. Long and T. Darrell, "Fully Convolutional Networks for Semantic Segmentation," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 4, pp. 640-651, 1 April 2017, doi: 10.1109/TPAMI.2016.2572683.

- Wang, W., Lin, H. & Wang, J. CNN based lane detection with instance segmentation in edge-cloud computing. J Cloud Comp 9, 27 (2020). https://doi.org/10.1186/s13677-020-00172-z

