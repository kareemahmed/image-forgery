# image-forgery
Authentic vs. Forged Image Classification Using ResNet50


This project focuses on building an image classification system capable of distinguishing **authentic** images from **forged** ones using deep learning. The dataset consists of two classes — *authentic* and *tampered* — prepared from Five different datasets: CASIA 2, Columbia, Comofod, MICC-F2000, and MICC-F200.

A two stream **ResNet50** convolutional neural network is used as the backbone model. The network is initialized with pretrained ImageNet weights to leverage transfer learning for faster convergence and improved accuracy on trained data. The first ResNet50 captures RGB image and the second ResNet50 captures FFT from each of the RGB channels. Training was made on 70% of each dataset then 15% is used for validation and 15% is used for testing. Evaluation metrics and results are presented for each dataset.

After training, the resulting model is able to accurately classify whether an image is authentic or forged. The trained model can then be integrated into applications for digital image forensics, document verification systems, or any automated image authenticity assessment pipeline.

to use this model run

python test_single_image.py <image_path>


