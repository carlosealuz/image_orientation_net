# Image Orientation Network
Exercise to use a neural network to detect the rotation of images.
Train dataset are images randomly rotated in one of the following perspectives:
  * Upright;
  * Rotated left;
  * Rotated right;
  * Upside down;

This dataset can be found here: dataset/train1.zip and **dataset/train2.zip.**
There is also a csv file with all images and their rotation.
The dataset used to evaluate the net is in **dataset/test.rotfaces.zip.**
The script to train and evaluate the network is in **scripts/cifar10_net.py**. You can run **python cifar10_net.py -d dataset/folder/path** in the console to train and evaluate the net.

I used the architecture [here](https://keras.io/examples/cifar10_cnn/) but I removed all Dropout layers to keep it simple.
Learning rate used was 0.001 and I splitted the training dataset as 80:20 (train, test images).

The code will save a csv file (**test.preds.csv**) with the predicted orientation for all images in validation dataset. 
The code will also save the best weights during training (using the validation loss as monitor). 

I used this csv with opencv to load and save all images with the upright orientation. You can check those images in this file **dataset/correct_orientation.zip**


