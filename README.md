# Universal-PS-CVPR2022
Official Pytorch Implementation of Universal Photometric Stereo Network using Global Lighting Contexts (CVPR2022)

```
S. Ikehata, "Universal Photometric Stereo Network using Global Contexts", CVPR2022
```

[project page](https://satoshi-ikehata.github.io/cvpr2022/univps_cvpr2022.html)

<p align="center">
<img src="webimg/top.png" width="800">  
</p>
<!-- <p align="center">
(top) Surface normal estimation result from 10 images, (bottom) ground truth.
</p> -->




## Getting Started

### Prerequisites

- Python3
- torch
- tensorboard
- cv2
- timm
- tqdm

Tested on:
- Windows11, Python 3.10.3, Pytorch 1.11.0, CUDA 11.3
  - GPU: Nvidia RTX A6000 (48GB)

### Prepare dataset
All training and test data must be formatted like this:

```bash
 Data
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

You can change the configuration (e.g., prefix, suffix) by source\modules\config

All masks in our datasets were computed using <a href="https://github.com/saic-vul/ritm_interactive_segmentation">software</a> by Konstantin.

### Download pretrained model 
Checkpoints of the network parameters (The full configuration in the paper) are available at <a href="https://www.dropbox.com/sh/pphprxqbayoljpn/AADUPNcAdOWkbGwRK6xo5Wura?dl=0">here</a> 

To use pretrained models, extract them as

```bash
  YOUR_CHECKPOINT_PATH
  ├── *.pytmodel
  ├── *.optimizer
  ├── *.scheduler
  └── ...

  ```

### Running the test
If you don't prepare dataset by yourself, please use some sample dataset from <a href="https://satoshi-ikehata.github.io/cvpr2022/univps_cvpr2022.html">here</a>

For running test, please run main.py as 

```
python source/main.py --session_name session_test  --mode Test --test_dir YOUR_DATA_PATH --pretrained YOUR_CHECKPOINT_PATH
```
Results will be put in ouput/session_name. 

### Running the training
The training script is NOT supported yet (will be available soon!).
However, the training dataset is alraedy available. Please send a request to sikehata@nii.ac.jp

## License
This project is licensed under the BSD License - see the [LICENSE.md](LICENSE.md) file for details