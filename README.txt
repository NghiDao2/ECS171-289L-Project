The source code for this project was written by Nghi Dao. Email: nmdao@ucdavis.edu
The code was also contributed to by Terry Tong. Email: tertong@ucdavis.edu

Make sure to run install.ipynb to install all the requirements as well as downloading the dataset

The dataset that we use is the Stanford cars dataset:
https://github.com/cyizhuo/Stanford-Cars-dataset

The implementation of the Mamba and Dit training is based on Dino Diffusion:
https://github.com/madebyollin/dino-diffusion

The implementation of the UNet Stable diffusion is based on this notebook:
https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing


The training notebook can be ran to train the model, but we also provided in the folder our trained model that can be loaded into the notebook.
The configurations of the pretrained models are the ones loaded into the notebook by default.
If you are using the pretrained models, make sure to load the following models:
pretrained_mamba_model.pth
pretrained_avg_mamba_model.pth
pretrained_transformer_model.pth
pretrained_avg_transformer_model.pth

The average models is a scaled average of all the models during training and provides more stable predictions while the normal model is simply the model after the last iteration of training

To reproduce similar (or better) results to our model, make sure to train all models for at least 6 hours on a GPU at least as powerful as the RTX 4090


Contributions to the project:
Nghi Dao: Wrote the source code, paper, ran the experiments, and made the video
Terry Tong: Contributed to the source code
Keenan Kalra: Edited the paper
