### 1. A man and a dog stand at opposite ends of a football field that is 100 feet long. Both start running towards each other.Let's say that the man runs at X ft/s and the dog runs at twice the speed of the man. Each time the dog reaches the man, the dog runs back to the end of the field where it started and then back to the man and repeat.What is the total distance the dog covers once the man finally reaches the end of the field?

A: 200 ft

 2 * 200 * (1⁄3+1⁄9+1⁄27+1⁄81+……) =200

The man traveled 100 feet, and the dog travels twice as fast, so the dog must have traveled 200 feet in the same time period (their start and end times are the same).

### 2. Metrics use to track accuracy and validity of the Classification model

a. ***Classification (confusion) metrics***

When performing classification predictions, there's four types of outcomes that could occur.
- True positives are when you predict an observation belongs to a class and it actually does belong to that class.
- True negatives are when you predict an observation does not belong to a class and it actually does not belong to that class.
- False positives occur when you predict an observation belongs to a class when in reality it does not.
- False negatives occur when you predict an observation does not belong to a class when in fact it does.

b. ***Accuracy, Precision, and Recall***

***Accuracy*** is defined as the percentage of correct predictions for the test data. It can be calculated easily by dividing the number of correct predictions by the number of total predictions.

Accuracy=correct predictions /all predictions

***Precision*** is defined as the fraction of relevant examples (true positives) among all of the examples which were predicted to belong in a certain class.

Precision=true positives / (true positives+false positives)

***Recall*** is defined as the fraction of examples which were predicted to belong to a class with respect to all of the examples that truly belong in the class.

Recall=true positives / (true positives+false negatives)

In cases where classes aren't evenly distributed,Recall ensures that we're not overlooking the '1', while precision ensures that we're not misclassifying too many '0' as '1'. 

c. ***Combined precision and recall: F-score***

Fβ=(1+β^2)*precision*recall / ((β^2⋅precision)+recall)

The β parameter allows us to control the tradeoff of importance between precision and recall. β<1 focuses more on precision while β>1 focuses more on recall.

d. ***Other practical advice***

Another common thing I'll do when evaluating classifier models is to reduce the dataset into two dimensions and then plot the observations and decision boundary. Sometimes it's helpful to visually inspect the data and your model when evaluating its performance.

### 3. Metrics use to track accuracy and validity of the Regression model

a. ***Explained variance*** compares the variance within the expected outcomes, and compares that to the variance in the error of our model. This metric essentially represents the amount of variation in the original dataset that our model is able to explain.

EV(ytrue,ypred)=1−Var(ytrue−ypred)/ytrue

b. ***Mean squared error*** is simply defined as the average of squared differences between the predicted output and the true output. Squared error is commonly used because it is agnostic to whether the prediction was too high or too low, it just reports that the prediction was incorrect.

MSE(ytrue,ypred)=(1/nsamples) * ∑(ytrue−ypred)^2

c. ***R2 coefficient*** represents the proportion of variance in the outcome that our model is capable of predicting based on its features.

R^2(ytrue,ypred)=1−∑(ytrue−ypred)^2/∑(ytrue−y¯)^2

### 4. Diagnose bias and variance 

a. ***Validation curves***

Tthe goal with any machine learning model is generalization. Validation curves allow us to find the sweet spot between underfitting and overfitting a model to build a model that generalizes well.

A typical validation curve is a plot of the model's error as a function of some model hyperparameter which controls the model's tendency to overfit or underfit the data. The parameter you choose depends on the specific model you're evaluating

b. ***Learning curves***

Learning curves is a plot of the model's error as a function of the number of training examples. Similar to validation curves, we'll plot the error for both the training data and validation data.

If our model has high bias, we'll observe fairly quick convergence to a high error for the validation and training datasets. If the model suffers from high bias, training on more data will do very little to improve the model. This is because models which underfit the data pay little attention to the data, so feeding in more data will be useless. A better approach to improving models which suffer from high bias is to consider adding additional features to the dataset so that the model can be more equipped to learn the proper relationships.

If our model has high variance, we'll see a gap between the training and validation error. This is because the model is performing well for the training data, since it has been overfit to that subset, and performs poorly for the validation data since it was not able to generalize the proper relationships. In this case, feeding more data during training can help improve the model's performance.






















