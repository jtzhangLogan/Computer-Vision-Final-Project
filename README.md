# Unpaired Image Style Transfer in Endoscopic Surgery 

This repository contains the source code for our final project "Unpaired Image Style Transfer in Endoscopic Surgery".  Please refer to our [final report](https://drive.google.com/file/d/1Z0Zaa2UG-rMAtpRcj-wrGyv4Xscq-rTs/view?usp=sharing) for detailed explanations. 

# Dataset
Train dataset is not open source. Please contact us to download. 

# Train

- Put the porcine model videos inside folder `input_data/train_1/Porcine/*`. Make sure you put videos performing different tasks (dissection, knot_tying, needle_passing) in the corresponding folder. 
- Put `get_frame.py` in each folder contains videos. Run this file, which will create the train video frames.
- Open `main.py`. Tune hyperparamaters of your choice. Make sure you pass `True` to argument `training` and  `False` to argument `testing`. 
- Run this file to start training. The trained parameter is stored in folder `checkpoints`. The file name format follows `generator-network_total-epoch_discriminator-network`.

# Test
- We have provided our pretrained checkpoint [here](https://drive.google.com/drive/folders/12Nk3yQhdNfCXDW-5uz6BB_IZAYIjYmOz?usp=sharing). Add this folder in your repository if you want to use our pretraied model.
- Put the VR simulation video frames in folder `test`. We have provided some video frames here. 
- Open `main.py`. Change the path to the `.ckpt` files in the `checkpoints` folder that you want to test. Make sure you pass `False` to argument `training` and  `True` to argument `testing`. 
- Run this file to start testing. The style transferred outputs are stored in folder `results`.

# Video Transfer
- Change the output picture in 'test.py' and make it only include 'y_fake_test'. Rerun the test.
- Open 'videotransfer.py'. Change the path to the certain place for the results and run it to get the video.


# Gatys
- Open 'Gatys.ipynb' and change the corresponding paths of content images and style images.
- Run it to get a set of transferred images.
- Then we can use the 'videotransfer.py' to get the result video.

# Example Video
- We use 'get_frame.py' to get 150 frames from one video in the original datasets and use these 150 frames to generate the example videos.
