February 27th, 2023 - Luke Atkins (lukeatki@iu.edu)

Build First Predictor - Multi-layer Perceptron

Log:

1. Downloaded the wine data set, set up my python venv for the project.

2. Normalized the wine data set with a python script. And printed the head of the data
so that I could see the head of the data, I then split the data into five random groups.

3. I created the Multi-layer perceptron class in a python file called MLP. I made its 
init function to set the number of hidden nodes, the data, and the weight vectors.

4. I re-read the the book to look for further guidance on how to set up my from scratch 
model, I choose an activation function of sigmoid for the hidden layer and softmac for the
output layer. what was challenging was thinking about the design of the class.

5. Finished the forwardpass algorithim, started on the backProp function, I suspect this to 
be the hardest of the functions. Train will be some iteration I suppose, we will see.

6. Went to math help for reading the notation of the back propogation algortithim, I found 
typo in the bookm the second layer weight should have another kappa index accosiated with 
them so for the second step of the error calculations in the hidden layer.

7. With this new found understanding of backpropogation, I went on to implement it in a 
manner that would be approved by the greatest of ML engineering and any obviously any 
hiring company at Open Ai. Whoo Hoo! Fiinished forward pas, finished getting the errors.

8. Fixed major bug that did not allow me to change the number of input node, lots of index
chasing to fix this one, was about to crush me, but I took a five min breather outside of 
rawls hall, came back, and fixed it lol. Thank goodness!

9. completyl finsihed backpropgation, I went with a Beta of 1 for the error function for 
simplicity. I hypothisis that this should not be felt in the accuracy of this model


