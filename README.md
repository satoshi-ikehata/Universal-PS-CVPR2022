# Universal-PS-CVPR2022
Official Pytorch Implementation of Universal Photometric Stereo Network using Global Lighting Contexts (CVPR2022)

```
[1] S. Ikehata, "Universal Photometric Stereo Network using Global Contexts", CVPR2022
```

[project page](https://www.bmvc2021-virtualconference.com/assets/papers/0319.pdf)

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

### Dataset Formart
All training and test data must be formatted like this:

```bash
  $ tree data
 Data
  ├── A [Suffix:default ".data"]
  │   ├── mask.png
  │   ├── [Prefix (default:"0" (Train), "L" (Test))] imgfile1
  │   ├── [Prefix (default:"0" (Train), "L" (Test))] imgfile2
  │   └── ...
  └── B [Suffix:default ".data"]
  │   ├── mask.png
  │   ├── [Prefix (default:"0" (Train), "L" (Test))] imgfile1
  │   ├── [Prefix (default:"0" (Train), "L" (Test))] imgfile2
  │   └── ...
   ```

### Running the test
Download the pretrained model from  <a href="https://www.dropbox.com/sh/pphprxqbayoljpn/AADUPNcAdOWkbGwRK6xo5Wura?dl=0">here</a>
e.g., YOUR_CHECKPOINT_PATH/***.pytmodel

Download the real dataset from <a href="https://satoshi-ikehata.github.io/cvpr2022/univps_cvpr2022.html">here</a>
e.g., YOUR_DATA_PTH/apple.data

Then, please run main.py as 

```
python source/main.py --session_name session_test  --mode Test --test_dir YOUR_DATA_PATH --pretrained YOUR_CHECKPOINT_PATH
```
Results will be put in ouput/session_name

You can change the test configuration by source\modules\config


### Important notice about DiLiGenT datasets

As is commonly known, "bear" dataset in DiLiGenT has problem and the first 20 images in bearPNG are skipped. 

### Running the test on othter datasets (Unsupported)
If you want to run this code on ohter datasets, please allocate your own data just in the same manner with DiLiGenT. The required files are
- images (.png format in default, but you can easily change the code for other formats)
- lights (light_directions.txt, light_intensities.txt)
- normals (normal.txt, if no ground truth surface normal is available, you can simply set all the values by zero)

### Running the training
The training script is NOT supported yet (will be available soon!).
However, the training dataset is alraedy available. Please send a request to sikehata@nii.ac.jp

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References
[2] Boxin Shi, Zhipeng Mo, Zhe Wu, Dinglong Duan, Sai-Kit Yeung, and Ping Tan, "A Benchmark Dataset and Evaluation for Non-Lambertian and Uncalibrated Photometric Stereo", In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2018.

### Comment
Honestly, the major reason why this work was aimed at "sparse" set-up is simply because the model size is huge and I didn't have sufficient gpu resources for training my model on "dense" iamges (though test on dense images using the model trained on sparse images is possible as shown in the paper).  I am confident that this model also benefits the dense photometric stereo task and if you have any ideas to reduce the training cost, they are very appreciated! 
