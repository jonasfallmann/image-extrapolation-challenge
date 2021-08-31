# Image Extrapolation Challenge 2021

Train a neural network to extrapolate unknown parts of an image.
- Samples considered in the challenge are grayscale images for which a certain amount of pixels at the borders of the images is unknown (=was set to zero).
- Your model should predict (=extrapolate) the unknown border pixel values.
- The images collected in exercise 1 will be the training set, however, you are free to include more images of your choosing into your training set. Since we already collected a lot of images, training on the training set itself will be sufficient.

Predict the unknown parts of the test set images
- You will be provided with test set images, where a certain amount of pixels at the borders of the images is unknown (=was set to zero).
- The test set images and the specifications of their unknown borders will be provided as pickle file (details in Assignment sheet 2).
- The test set images will have a shape of (90, 90) pixels.
- The area of known pixels in the test set images will be at least (75, 75) pixels large.
- The borders containing unknown pixels in the test set images will be at least 5 pixels wide on each side.

# Sample output
![alt text](https://github.com/jonasfallmann/image-extrapolation-challenge/tree/main/sample/0000016_00.png "Sample network ouput")