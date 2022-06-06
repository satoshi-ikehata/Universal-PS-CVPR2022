# Universal-PS-CVPR2022
Official Pytorch Implementation of Universal Photometric Stereo Network using Global Lighting Contexts (CVPR2022)

```
Satoshi Ikehata, "Universal Photometric Stereo Network using Global Contexts", CVPR2022
```
<p align="center">
<a href="https://satoshi-ikehata.github.io/cvpr2022/univps_cvpr2022.html"> <img src="webimg/top.png" width="800", alt="project site">  </a>
</p>
<!-- <p align="center">
(top) Surface normal estimation result from 10 images, (bottom) ground truth.
</p> -->

## Prerequisites
- Python3
- torch
- tensorboard
- cv2
- timm
- tqdm

Tested on:
- Windows11, Python 3.10.3, Pytorch 1.11.0, CUDA 11.3
  - GPU: Nvidia RTX A6000 (48GB)

## Prepare dataset
All you need for running the universal photometric stereo network is shading images and a binary object mask. The object could be illuminated under arbitrary lighting sources but shading variations should be sufficient (weak shading variations may result in poor results).

In my implementation, all training and test data must be formatted like this:

```bash
 YOUR_DATA_PATH
  ├── A [Suffix:default ".data"]
  │   ├── mask.png
  │   ├── [Prefix (default:"0" (Train), "L" (Test))] imgfile1
  │   ├── [Prefix (default:"0" (Train), "L" (Test))] imgfile2
  │   └── ...
  └── B [Suffix:default ".data"]
      ├── mask.png
      ├── [Prefix (default:"0" (Train), "L" (Test))] imgfile1
      ├── [Prefix (default:"0" (Train), "L" (Test))] imgfile2
      └── ...
  ```

For more details, please see my real dataset at <a href="https://satoshi-ikehata.github.io/cvpr2022/univps_cvpr2022.html">project page</a>.
You can change the configuration (e.g., prefix, suffix) at <a href="https://github.com/satoshi-ikehata/Universal-PS-CVPR2022/tree/main/source/modules/config.py">source\modules\config.py</a>.


All masks in our datasets were computed using <a href="https://github.com/saic-vul/ritm_interactive_segmentation">the software by Konstantin</a>.

## Download pretrained model 
Checkpoints of the network parameters (The full configuration in the paper) are available at <a href="https://www.dropbox.com/sh/pphprxqbayoljpn/AADUPNcAdOWkbGwRK6xo5Wura?dl=0">here</a> 

To use pretrained models, extract them as

```bash
  YOUR_CHECKPOINT_PATH
  ├── *.pytmodel
  ├── *.optimizer
  ├── *.scheduler
  └── ...

  ```

## Running the test
If you don't prepare dataset by yourself, please use some sample dataset from <a href="https://satoshi-ikehata.github.io/cvpr2022/univps_cvpr2022.html">here</a>

For running test, please run main.py as 

```
python source/main.py --session_name session_test --mode Test --test_dir YOUR_DATA_PATH --pretrained YOUR_CHECKPOINT_PATH
```
Results will be put in ouput/session_name. 

## Running the training
For running training, please run main.py as:
```
python source/main.py --session_name session_train --mode Train --training_dir YOUR_DATA_PATH
```
or if you want to perform both training and test, instead use this:

```
python source/main.py --session_name session_train_test --mode TrainAndTst --training_dir YOUR_DATA_PATH --test_dir YOUR_DATA_PATH
```

The default hyperparameters are described in <a href="https://github.com/satoshi-ikehata/Universal-PS-CVPR2022/tree/main/source/main.py">source/main.py</a>.

The trainind data (PS-Wild) will be distributed before CVPR2022.

## License
This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details