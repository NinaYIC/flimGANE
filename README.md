# *flimGANE*
<u>**f**</u>luorescence **l**ifetime **im**aging method based on **G**enerative **A**dversarial **N**etwork **E**stimation

This code implements the model from the paper "[Generative adversarial network enables rapid and robust fluorescence lifetime image analysis in live cells](https://www.nature.com/articles/s42003-021-02938-w)". It is a new deep learning-based fluorescence lifetime estimation method based on Generative Adversarial Network framework. *flimGANE* can rapidly generate accurate and high-quality FLIM images even in the photon-starved conditions. We demonstrated our model is not only faster but also provide more accurate analysis in barcode identification, cellular structure visualization, Förster resonance energy transfer characterization, and metabolic state analysis. With its advantages in speed and reliability, *flimGANE* is particularly useful in fundamental biological research and clinical applications, where ultrafast analysis is critical.

Prerequisites
----------------------------------------------------

The code is written in Python 3.6.5. You will also need:
- **TensorFlow** version 1.12.0 or above

Getting started
----------------------------------------------------

Before starting, you need to generate the Monte Carlo simulation dataset. The example simulation dataset can be found [here](10.6084/m9.figshare.19086368)! 

Training a *flimGANE* model
----------------------------------------------------

Now given we have the training datasets, we can train the models. 

Generate FLIM images using trained *flimGANE* model
----------------------------------------------------

Once your model is finished training, the next step is to implement it. 

Contact
----------------------------------------------------

File any issues with the [issue tracker](https://github.com/NinaYIC/flimGANE/issues). For acy questions or problems, this code is maintained by [@NinaYIC](https://github.com/NinaYIC).

## Reference

- Yuan-I Chen, Yin-Jui Chang, Shih-Chu Liao, Trung Duc Nguyen, Jianchen Yang, Yu-An Kuo, Soonwoo Hong, Yen-Liang Liu, H. Grady Rylander III, Samantha R. Santacruz, Thomas E. Yankeelov & Hsin-Chih Yeh (2022). [Generative adversarial network enables rapid and robust fluorescence lifetime image analysis in live cells](https://www.nature.com/articles/s42003-021-02938-w), *Communications Biology*.
