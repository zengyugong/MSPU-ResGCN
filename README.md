# MSPU-ResGCN: Multi-Scale Point Cloud Upsampling using Residual Graph Convolutional Networks 


### Update
* 2025/04/04: First submition


### Preparation

1. Clone the repository:

   ```shell
   git clone https://github.com/zengyugong/MSPU-ResGCN.git
   cd PU-GCN
   ```
   
2. install the environment
   Once you have modified the path in `compile.sh` under `tf_ops`, you can simply install `pugcn` environment by:  
   
   ```bash
    bash env_install.sh
    conda activate pugcn
   ```
   
   Note this repository is based on Tensorflow (1.13.1) and the TF operators from PointNet++.  You can check the `env_install.sh` for details how to install the environment.  In the second step, for compiling TF operators, please check `compile.sh` in `tf_ops` folder, one may have to manually change the path!!
   
3. A seperated dataset uploaded in the /data/train folder
   
4. you can download full PU1K dataset from [Google Drive](https://drive.google.com/file/d/1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg/view?usp=sharing)  
    Link the data to `./data`:

    ```bash
    mkdir data
    ln -s /path/to/PU1K ./data/
    ```
5. Optional. The original meshes of PU1K dataset is avaialble in [Goolge Drive](https://drive.google.com/file/d/1tnMjJUeh1e27mCRSNmICwGCQDl20mFae/view?usp=sharing)
    
### Train on PU1K (Random input) 

**note**: If you favor uniform inputs, you have to retrain all models. Otherwise, the results might be really bad. To train with uniform inputs, simply add `--fps` in the command line below.
We provide the **pretrained PU-GCN on PU-GAN's dataset using the uniform inputs** [here](https://drive.google.com/file/d/1xdG3hUomPoUhdusuYjHqCyl8YMBwYrZg/view?usp=share_link) in case it is needed. 

To train models on PU1K using **random inputs**. Our pretrained models (PU-GCN on PU1K random and other models) are available [Google Drive](https://drive.google.com/file/d/1vusBIw7sd69gnyaeoWMiGaPHfkyHM5Qb/view?usp=sharing)

To train on other dataset, simply change the `--data_dir` to locate to your data file. 

-  MSPU-ResGCN
    ```shell
    python main.py --phase train --model pugcn_residual --upsampler nodeshuffle --k 20 --data_dir data/train/poisson_1024_data_part_1.h5 --block inception_ms
    ```
  if use 256 point cloud data
   ```shell
   python main.py --phase train --model pugcn_residual --upsampler nodeshuffle --k 20 --data_dir data/train/poisson_256_data_part_1.h5 --up_ratio 1 --block inception_ms
   ```
-  PU-GCN
    ```shell
    python main.py --phase train --model pugcn --upsampler nodeshuffle --k 20 --k 20 --data_dir data/train/poisson_1024_data_part_1.h5 --block inception
    ```

-  PU-Net
    ```
    python main.py --phase train --model punet --upsampler original  
    ```

-  MPU
    ```
    python main.py --phase train --model mpu --upsampler duplicate 
    ```

-  PU-GAN
    ```
    python main.py --phase train --model pugan --more_up 2 
    ```



### Evaluation

1. Test on PU1K dataset
   ```bash
   bash test_pu1k_allmodels.sh # please modify this script and `test_pu1k.sh` if needed
   ```

5. Test on real-scanned dataset

    ```bash
    bash test_realscan_allmodels.sh
    ```

6. Visualization. 
    check below. You have to modify the path inside. 
    
    ```bash
    python vis_benchmark.py
    ```
    



## Citation



​    
### Acknowledgement
This repo is heavily built on [PU-GAN code](https://github.com/liruihui/PU-GAN). We also borrow the architecture and evaluation codes from PUGCN, MPU and PU-Net. 


