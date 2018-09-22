# LSC and DIP

**Author**: Martino Ferrari   
**Contact**: manda.mgf@gmail.com

## Presentation

My master thesis work extends the problem formulation of learnable compressive subsampling [1] that focuses on the learning of the best sampling operator in the Fourier domain adapted to spectral properties of a training set of images. I formulated the problem as a reconstruction from a finite number of sparse samples with a prior learned from the external dataset or learned on-fly from the images to be reconstructed. More in 
details, I developed two very different methods, one using multiband coding in the spectral domain and the second using a neural network. 

The new methods can be applied to many different fields of spectroscopy and Fourier optics, for example in medical (computerized tomography, magnetic resonance spectroscopy) and astronomy (the Square Kilometre Array) imaging, where the capability to reconstruct high-quality images, in the pixel domain, from a limited number of samples, in the frequency domain, is a key issue. 

The proposed methods have been tested on diverse datasets covering facial images, medical and multi-band astronomical data, using the mean square error and SSIM as a perceptual measure of the quality of the reconstruction. 

Finally, I explored the possible application in data acquisition systems such as computer tomography and radio astronomy. The obtained results demostrate that the properties of the proposed methods have a very promising potential for future research and extensions. 

For such reason, the work was both presented at EUSIPCO 2018 conference [2] ([pdf](http://sip.unige.ch/articles/2018/EUSIPCO2018_TaranO.pdf)) in Rome and submitted for a EU patent. 

## Results

More raw results can be found on the `results` folder and other can be created by running the different scripts.


## Notebooks

Two example notebooks can be found in the folder `dip` and `lsc`.

## Execute

To execute `lsc` or `dip` simply execute the following:
```
python3 dip conf.yml
```
or 
```
python3 lsc conf.yml
```

## Configurations

Some configuration examples are stored in `lsc_conf` and `dip_conf`.

----
[1]: L. Baldassarre, Y.-H. Li, J. Scarlett, B. Gözcü, I. Bogunovic, and V. Cevher, “Learning-based compressive subsampling,” IEEE Journal of Selected Topics in Signal Processing, vol. 10, no. 4, pp. 809–822, 2016

[2]: M. Ferrari, O. Taran, T. Holotyak, K. Egiazarian, and S. Voloshynovskiy, "Injecting Image Priors into Learnable Compressive Subsampling," in Proc. 26th European Signal Processing Conference (EUSIPCO), Rome, Italy, 2018
