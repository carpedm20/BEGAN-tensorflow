# BEGAN in Tensorflow

Tensorflow implementation of [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717).

![alt tag](./assets/model.png)


## Requirements

- Python 2.7
- [Pillow](https://pillow.readthedocs.io/en/4.0.x/)
- [tqdm](https://github.com/tqdm/tqdm)
- [requests](https://github.com/kennethreitz/requests) (Only used for downloading CelebA dataset)
- [TensorFlow 1.1.0](https://github.com/tensorflow/tensorflow) (**Need nightly build** which can be found in [here](https://github.com/tensorflow/tensorflow#installation), if not you'll see `ValueError: 'image' must be three-dimensional.`)


## Usage

First download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets with:

    $ apt-get install p7zip-full # ubuntu
    $ brew install p7zip # Mac
    $ python download.py

or you can use your own dataset by placing images like:

    data
    └── YOUR_DATASET_NAME
        ├── xxx.jpg (name doesn't matter)
        ├── yyy.jpg
        └── ...

To train a model:

    $ python main.py --dataset=CelebA --use_gpu=True
    $ python main.py --dataset=YOUR_DATASET_NAME --use_gpu=True

To test a model (use your `load_path`):

    $ python main.py --dataset=CelebA --load_path=CelebA_0405_124806 --use_gpu=True --is_train=False --split valid


## Results

### Autoencoded Generator outputs

<img src="./assets/AE_G1.png" width="7%"> <img src="./assets/AE_G2.png" width="7%">
<img src="./assets/AE_G3.png" width="7%"> <img src="./assets/AE_G4.png" width="7%">
<img src="./assets/AE_G5.png" width="7%"> <img src="./assets/AE_G6.png" width="7%">
<img src="./assets/AE_G7.png" width="7%"> <img src="./assets/AE_G8.png" width="7%">
<img src="./assets/AE_G9.png" width="7%"> <img src="./assets/AE_G10.png" width="7%">
<img src="./assets/AE_G11.png" width="7%"> <img src="./assets/AE_G12.png" width="7%">
<img src="./assets/AE_G13.png" width="7%"> <img src="./assets/AE_G14.png" width="7%">
<img src="./assets/AE_G15.png" width="7%"> <img src="./assets/AE_G16.png" width="7%">
<img src="./assets/AE_G17.png" width="7%"> <img src="./assets/AE_G18.png" width="7%">
<img src="./assets/AE_G19.png" width="7%"> <img src="./assets/AE_G20.png" width="7%">
<img src="./assets/AE_G21.png" width="7%"> <img src="./assets/AE_G22.png" width="7%">
<img src="./assets/AE_G23.png" width="7%"> <img src="./assets/AE_G24.png" width="7%">
<img src="./assets/AE_G25.png" width="7%"> <img src="./assets/AE_G26.png" width="7%">


### Interpolation of real images with Discriminator

![alt tag](./assets/AE_batch.png)
![alt tag](./assets/interp_1.png)
![alt tag](./assets/interp_2.png)
![alt tag](./assets/interp_3.png)
![alt tag](./assets/interp_4.png)
![alt tag](./assets/interp_5.png)
![alt tag](./assets/interp_6.png)
![alt tag](./assets/interp_7.png)
![alt tag](./assets/interp_8.png)
![alt tag](./assets/interp_9.png)
![alt tag](./assets/interp_10.png)


### Generator and Discriminator outputs (after 58k step)

<img src="./assets/58600.png" width="80%">

(in progress)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
