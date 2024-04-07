# Dino-Nuggetology
UCSD 2024 DataHacks Entry
* Team Name: Decaffeinated
* Team Members: So Hirota, Penny King, Garvey Li

## Project Premise 
We created an image multi-classification classifier for dinosaur chicken nuggets. Photos of dino nuggets are classified into "normal" or "deformed" groups. "Normal" dino nuggets are then classified into their respective dinosaur species.

Our model has potential use cases in product quality detection, specifically in catching less than ideal products before selling. This can help increase customer satisfaction. Our image classification model can also be applied in anomaly detection more generally for various other cases


### Data
All of our data we collected ourselves using the “Dino Buddies” brand. In total there are 7 dinosaur shapes which are T-rex, Triceratops, Pterodactyl, Stegosaurus, Brontosaurus, and Parasaurolophus. And there is also an additional shape we created for the purposes of our assignment –– the “deformed” shape.


We took over 200 photos of various dino nugies on a white paper plate all from the same distance from the camera. We created additional data points by mirroring photos and rotating our photos in increments of 15 degrees. This allows our model to generalize to various orientations. In total this resulted in over 10 thousand photos in our dataset.


Preprocessing our data consisted of the following steps:

We converted our RGB jpeg images into grayscale images with values from 0-255 and reduced image size to 200 x 200 using bicubic interpolation in order to standardize our images.

To create an image containing only edge information we first used a gaussian low pass filter to blur the image to mitigate any potential noise (such as stray bread crumbs or the textures of the dino nuggets). We then calculated the 1st order gradient image in the x and y directions by convolving the image with two sobel kernels. With the gradient image, we further thresholded the image to make it black and white, resulting in a clean outline of the nugget.
 
With this initial set of data, we decided to create more images by augmenting the preexisting images. So, for each image, we rotated it in 15 degree increments and created a mirrored version that was also rotated, thus creating 23 new images or data points per sample.



### Model

For our model, we decided on a convolutional autoencoder. This is because an autoencoder is good at learning low dimensional representation of images. The idea is to train a model to learn representation of normal nuggets, so it will fail to reconstruct deformed nuggets. We set a threshold of error for reconstruction to detect if a nugget is deformed or not.

The encoder and decoder consist of 3 convolution layers and 1 dense layer. The convolution layers help reduce the dimension of the input, and the dense layer flattens the image into a 3000 row latent representation. For the activations in the intermediate layers, we used ReLU, and for the final output, used sigmoid. This is to force the output values to range from 0 to 1, which is how the images are represented.. 


