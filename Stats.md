### 1. Metrics use to track accuracy and validity of the Classification model

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

### 2. Metrics use to track accuracy and validity of the Regression model

a. ***Explained variance*** compares the variance within the expected outcomes, and compares that to the variance in the error of our model. This metric essentially represents the amount of variation in the original dataset that our model is able to explain.

EV(ytrue,ypred)=1−Var(ytrue−ypred)/ytrue

b. ***Mean squared error*** is simply defined as the average of squared differences between the predicted output and the true output. Squared error is commonly used because it is agnostic to whether the prediction was too high or too low, it just reports that the prediction was incorrect.

MSE(ytrue,ypred)=(1/nsamples) * ∑(ytrue−ypred)^2

c. ***R2 coefficient*** represents the proportion of variance in the outcome that our model is capable of predicting based on its features.

R^2(ytrue,ypred)=1−∑(ytrue−ypred)^2/∑(ytrue−y¯)^2

### 3. Diagnose bias and variance 

a. ***Validation curves***

Tthe goal with any machine learning model is generalization. Validation curves allow us to find the sweet spot between underfitting and overfitting a model to build a model that generalizes well.

A typical validation curve is a plot of the model's error as a function of some model hyperparameter which controls the model's tendency to overfit or underfit the data. The parameter you choose depends on the specific model you're evaluating

b. ***Learning curves***

Learning curves is a plot of the model's error as a function of the number of training examples. Similar to validation curves, we'll plot the error for both the training data and validation data.

If our model has high bias, we'll observe fairly quick convergence to a high error for the validation and training datasets. If the model suffers from high bias, training on more data will do very little to improve the model. This is because models which underfit the data pay little attention to the data, so feeding in more data will be useless. A better approach to improving models which suffer from high bias is to consider adding additional features to the dataset so that the model can be more equipped to learn the proper relationships.

If our model has high variance, we'll see a gap between the training and validation error. This is because the model is performing well for the training data, since it has been overfit to that subset, and performs poorly for the validation data since it was not able to generalize the proper relationships. In this case, feeding more data during training can help improve the model's performance.

### 4. What are MLE and MAP? What is the difference between the two?

Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP), are both a method for estimating some variable in the setting of probability distributions or graphical models. They are similar, as they compute a single estimate, instead of a full distribution.

MLE gives you the value which maximises the Likelihood P(D|θ). And MAP gives you the value which maximises the posterior probability P(θ|D). As both methods give you a single fixed value, they’re considered as point estimators.

Both Maximum Likelihood Estimation (MLE) and Maximum A Posterior (MAP) are used to estimate parameters for a distribution. MLE is also widely used to estimate the parameters for a Machine Learning model, including Naïve Bayes and Logistic regression. It is so common and popular that sometimes people use MLE even without knowing much of it. For example, when fitting a Normal distribution to the dataset, people can immediately calculate sample mean and variance, and take them as the parameters of the distribution. 

Using the formula, first we need to derive the log likelihood function, then maximize it by making a derivative equal to 0 with regard of Θ or by using various optimization algorithms such as Gradient Descent. Because of duality, maximize a log likelihood function equals to minimize a negative log likelihood. In Machine Learning, minimizing negative log likelihood is preferred. For example, it is used as loss function, cross entropy, in the Logistic Regression.

Based on the formula above, we can conclude that MLE is a special case of MAP, when prior follows a uniform distribution. This is the connection between MAP and MLE.

[link][https://towardsdatascience.com/mle-vs-map-a989f423ae5c]

### 5. How do you detect and handle correlation between variables in linear regression? What will happen if you ignore the correlation in the regression model?
A goal of regression analysis is to isolate the relationship between each independent variable and the dependent variable.when independent variables are correlated, it indicates that changes in one variable are associated with shifts in another variable. The stronger the correlation, the more difficult it is to change one variable without changing another. It becomes difficult for the model to estimate the relationship between each independent variable and the dependent variable independently because the independent variables tend to change in unison. two problem can occur: 1.The coefficient estimates can swing wildly based on which other independent variables are in the model. The coefficients become very sensitive to small changes in the model. 2. Multi col-linearity reduces the precision of the estimate coefficients, which weakens the statistical power of your regression model. You might not be able to trust the p-values to identify independent variables that are statistically significant.

There is a very simple test to assess multicollinearity in your regression model. The variance inflation factor (VIF) identifies correlation between independent variables and the strength of that correlation. Statistical software calculates a VIF for each independent variable. VIFs start at 1 and have no upper limit. A value of 1 indicates that there is no correlation between this independent variable and any others. VIFs between 1 and 5 suggest that there is a moderate correlation, but it is not severe enough to warrant corrective measures. VIFs greater than 5 represent critical levels of multicollinearity where the coefficients are poorly estimated, and the p-values are questionable.













