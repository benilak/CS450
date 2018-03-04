Neural net project from CS450 - Machine Learning

(See 'neuralnet.py' for the primary code containing the classifier and model).

The NetClassifier uses the Multi-Layer Perceptron algorithm to build a neural net, which can later be used to make predictions given a set of testing data. 

The learning plots show the number of correctly "predicted" targets for each training cycle on the y-axis, with the cycle number (or time) on the x-axis. This gives an idea of the accuracy of the model as it continues to train on the data. Interestingly, the plots often showed accuracy levels approaching 90% or greater, but the predictions from the final model were usually much less accurate. This could be due to overfitting, or simply because the testing data contains new information that was never trained on.
