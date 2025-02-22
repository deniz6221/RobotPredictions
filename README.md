## Robot Final State Prediction Using Several Models

This project uses 3 three different models to predict the final state of a robot arm. \
It predicts the result by inputting the top-down image of the robot at t=0 and the action its going to take.

### MLP Model
The MLP model is simple, it flattens the image and the image goes through several linear layers. After that it concats the action id and predicts the final position. \
However the neuron count is too high, the model is quite large because of this (202 MB). And its predictions are not great since plain MLP isn't the way to process images. \
Loss plot can be found in MLPLoss.png, code is present in MLPNN.ipynb

### CNN Model
The CNN model uses convolutional layers to process the image at first, then it flatens the output and concats the action id. After going through several linear layers, it predicts the output. \
This model predicts the output better and its size is significantly lower. \
Loss plot can be found in CNNLoss.png, code is present in CNN.ipynb

### Reconstruction Model
This model uses convolutional layers and transpose convolutional layers to construct an image of the robots final state. \
The initial image goes through convolutional layers at first, then it gets flattened and concatanated with the action id. After that it goes through several linear layers. \
At last transpose convolutional layers construct an image from the linear output. \
The model predicts some image (ReconstructedImages.png) but its not perfect, the code can be found in Reconstruction.ipynb

### Conclusion

I only trained the models for 10000 iterations. Although they output okey results they can significantly improve with more training.
