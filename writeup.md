# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lauryndbrown/TrafficSignClassifier/blob/master/src/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


See Step 1: Dataset Summary & Exploration in the code for information on size of training set, validation set, and test set.


#### 2. Include an exploratory visualization of the dataset.

See Step 1: Dataset Summary & Exploration in the code for visualizations of the images and each unqiue class.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to  augment the data because there were some classes that did not have much representation in the dataset and I was concerned that I wouldn't have enough for my model to learn those signs. To augment the data, I alter brightness, blur, translation, and rotation by small amounts that will not affect the meaning of the sign. This is particularily important for rotation.

See the section in the code labeled `Augment Data` for a more in depth look at a visualization of each augmentation function as well as how the numbers of images and distributions of those images changed. 


To preprocess the images, I first resize the images. While, it's not important for this portion of the project, it is necessary in a later section. I then preform histogram equalization, to aid in being able to identity those images that are of poorer quality (poor brightness and/or contrast). I convert the images to grayscale because in this dataset the color doesn't matter when identitifying the sign, and would cause more difficulty to train the model due to variance in colors and lighting. Finally, I normalize the data.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

See the section labeled `Model Architecture` in my code, for information on the model as well as a table displaying information aboout each layer.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an `tf.train.AdamOptimizer` with a rate of `0.0009`. I run 13 Epochs with a batch size of 100. I also tuned the dropout parameter to 0.4. Training the model was a matter of trial and error. I would change each parameter at a time and see what the results were. I have notice that dropout for my model was the variable with single largest effect.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.960 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? I initially chose the Lenet architecture from class. It seemed like a solid starting place for the problem.
* What were some problems with the initial architecture? I couldn't get much better than a little less than 90% with the Lenet architecture by simply tuning the parameters.I felt that I needed something a bit more complex in order to get to the 93% or more.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. I added another convolution layer. Like the paper in the suggested reading for this project, I decided to combine the last two convolutional layers. This resulted in a boost of accuracy. Outside of that, adding an aggressive drop value 0.4 was key in getting the accuracy up.
* Which parameters were tuned? How were they adjusted and why? As stated above, I tuned learning rate, batch size, number of epochs, and dropout value. The learning rate and batch size was adjusted down to 0.0009 and 100 respectively. I adjusted the number of epochs slightly as there are times when it appeared to improve past 10 epocs. I experiented wiht the dropout value a lot due to the fact that it had the most dramatic changes. I found 0.4 to be the best. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? Convolutional layers appear to work well for a variety of image classification problems. While the lenet lab only showed identifying black and white handwritten numbers, that architecture can be applied to a variety of problems. Dropout becomes useful when you find yourself overfitting the data. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose Do not enter, pedestrian, turn left, 70km speed sign, and stop sign. See `Step 3: Test a Model on New Images` for a visualization of the signs I chose. 

These might be more difficult because:
* Do not enter, pedestrian, 70km speed sign, and stop sign contained backgrounds of trees, sky, buildings, and roads. 
* The pedestrian sign had been cropped so that all of it's edges cannot be seen in th photo.
* The images found contained more intense color than those in the dataset. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Pedestrian     			| U-turn 										|
| 70km speed sign					| Yield											|
| Turn Left     		| Bumpy Road					 				|
| Do not enter		| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| Pedestrian										|
| .05					| 70km speed sign											|
| .04	      			| Turn Left				 				|
| .01				    | Do not enter      							|


For the second image ... 
For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


