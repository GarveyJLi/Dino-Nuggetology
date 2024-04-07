# datahacks_2024
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


We converted our images to grayscale and reduced image size to 200 x 200 using bicubic interpolation. We then used a Gaussian filter to blur the images which reduces fine grain details… such as crumbs and contours on the nugget. Then we used sobel kernels to find the image gradient. This results in an image containing the edges of the nugget as seen on the far right. After get gradient image, we finally thresholded to make the image binary, and this results in a black and white only image. The resulting data is a 2D array of 0s and 1s corresponding to the pixels in our image. We used these arrays for model training and testing.


### Model


