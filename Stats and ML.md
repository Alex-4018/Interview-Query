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

### 6. You want to determine a relationship between two variables.What is the downside of only using the R-Squared (R2) value to do so?

R-squared is a goodness-of-fit measure for linear regression models. This statistic indicates the percentage of the variance in the dependent variable that the independent variables explain collectively. R-squared measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale.
You cannot use R-squared to determine whether the coefficient estimates and predictions are biased, which is why you must assess the residual plots.

Overfitting, not correcting for model complexity, doesn’t tell which variable most important, doesn’t tell the error bar (uncertainty) on each variable.

### 7. How would you tackle multicollinearity in multiple linear regression?

I would suggest it is better to draw a correlation matrix, or heat map to illustrate the correlation between any two factors. Only when all non-diagonal elements are smaller than a given threshold should they be considered independent.

If there are unsatisfied elements, available options include: PCA, which is supposed to reduce dimensionality, is also helpful in minimizing the correlationship;,Ridge/Lasso Regression, which punishes the biases from models; obtain more data, so as to reduce variance.

You could use the Variance Inflation Factors (VIF) to determine if there is any multicollinearity between independent variables — a standard benchmark is that if the VIF is greater than 5 then multicollinearity exists.

Multicollinearity undermines the statistical significance of an independent variable. While it may not necessarily have a large impact on the model’s accuracy, it affects the variance of the prediction and reduces the quality of the interpretation of the independent variables

### 8. Between linear regression and random forest regression, which model would perform better and why?

A linear regression is a linear model. Random forest is a tree-based model that grows trees in parallel. These are two completely different models so there are a lot of differences:
Linear Regression has many assumptions 1) normal distribution of error terms 2) independence of the predictors 3) mean residuals = 0 and constant variance 4) no multi-collinearity or autocorrelation. Random Forest, on the other hand, does not have these assumption requirements.
Linear Regression cannot handle cardinality and can be affected by extreme outliers. Random Forest handles missing values and cardinality very well and is not influenced by extreme outliers
A linear regression will work better if the underlying distribution is linear and has many continuous predictors. Random Forest will tend to be better with categorical predictors.
Both will give some semblance of a “feature importance.” However, linear regression feature importance is much more interpretable
Usually random forest regressor being an ensemble technique should be better than linear regression. If the booking prices follow a linear trend, or if there are a large number of features, it may be more performance-friendly to use linear regression. In general, random forest will outperform linear regression. But as mentioned before, it’ll depend on the distribution of the dataset and the different available predictors. There is no cookie cutter, “ this model always performs better.”

### 9. Let's say you have a categorical variable with thousands of distinct values, how would you encode it?
I would go with Target Encoding. Encode the mean of the target variable as the value for that category.
If we take a neural network approach then Entity Embeddings can also be used. In this we train an embedding matrix for the category, so like in text data we have embeddings for each word, out categories will each have an embedding for them. It will allow the neural network to learn a better representation of that category and the dimensionality would also be less.

Basic Target Encoding - Encode the mean of the target value as the feature value. The benefit is that:
Reduce the need to one hot encode, so we reduce the dimensionality of our feature set. This makes using this feature on simple regressions more palatable.
KFold Target Encoding - a “regularized” version, to prevent overfitting onto our train set. This ensures that our feature will generalize better to unseen data.
Label Encoding - We can just convert these distinct values to a specific label. However, I like target encoding better, because it’ll incorporate information about the magnitude of its effect on the target. With label encoding, we’re treating the distance between each label equally.
1. One-hot encoding 
    * pros - two levels numeric encoding , naive 
    * cons - given the cardinality, dimensions increase, curse of dimensionlity 
2. label encoding 
    * pros - numeric vector given the categories 
    * cons - does not add any predictive power, confusion around the interpretability 
3. Count encoding 
    * pros - indicative of the frequency, predictive power 
    * cons - 
4. Target encoding - proportion/avg of Y for this categorical level 
    * pros - predictive power, better performance , learning from the labels 
    * cons - chances of target leakage if not implemented properly 
5. Cat Boost encoding - similar to target encoding but takes proportion will the current row in a sequential manner 
6. WOE - binning = WOE(Ci) = log(+ve examples asscicated with that category /-ve examples associated with the category)
    * gives the score of how representative the cat level w.r.t the postive label 
    * pros - best performance 
    * cons - target leakage if not implemented correctly
    
### 10. Bank Fraud Model
Training Data (chargebacks due to fraud & historical fraud txns)
If imbalanced data, then use the resampling technique to boost the fraud example
Model perspective (Use a simple Logistic Regression / RandomForest based so that realtime inference is fast). We could also explore Ensemble of multiple models (if runtime perspective if that is fast enough)

A simple classification model emitting probability score (a threshold could be chosen) to take the decision

Features:
POS / web/ phone
Time (temporal features)
Realtime features (location)
txn_amount
txn_currency
Card in person
Was the card stolen or on hold
Is there a overdraft protection
Is txn_amount > 500 USD
num of times card used in the previous month
num of average txn amount for the last 90 days
is_international card / txn ?
Does it involve multi-currency ?
Is this from a different location than the correct Address ?
Cost consideration (False Positive vs False Negative)

Measure based on the cost consideration (As this could be highly class imbalanced data, we could consider F1-score along with accuracy and AUC)


Fraudulent data are imbalanced data, which means TP+FN is very small compared to TN+FP.
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
Accuracy= (TP+TN)/(TP+TN+FP+FN)
For an imbalance dataset, we should use Precision or Recall instead of Accuracy.

From business/practical perspective, what are the consequences for FP or FN?
FP: customers receive false-alarm texts and get annoyed
FN: fraud gets undetected; huge loss to the bank and customers
Therefore, the model should focus on minimizing FN(type 2 error)

### 11. What methods could you use to increase recall?
Recall is how many of the true valid searches that the search algorithm gets right. It is TP/(TP+FN). You can increase recall by decreasing the FN or the False Negative. If you lower the threshold (in extreme case threshold is 0 and everything is classified as a valid search so FN will be 0) used by the search algorithm then number of false negatives will go down but False Positives (FP) will also go up. Looking at precision-recall curve can help identify this tradeoff between recall and precision.
Using ROC curve technique for current search algorithm, by changing the model hyper parameters throughout their range will give idea about at what combination of model hyper parameters, the current algorithm yields best results in terms of TPR and FPR. Minimum FPR and maximum TPR is goal to get true positive results and reduce false positives.

### 12. Assumption of Linear Regression
Assumptions of linear regression:
- Linear relationship between X and Y
- Error terms are normally distributed,Error term has a mean of 0
- Error terms are independent of each other
- Error terms have constant variation
- Low or no correlation between any two variables

### 13. What are kernel methods in machine learning? What are the requirements for a matrix to represent a kernel? What happens if we run a support vector machine model using a kernel that does not satisfy these requirements?
Kernel methods ( implicitly) map a linear classifier to a non-linear space. In layman’s terms, they let you look for hyperplanes in different places.
Kernel matrix needs to be positive semi-definite. (This is a concept from linear algebra, covered later in the semester in an intro course.)

positive semi-definite matrices provide a metric to your space, if the matrix is not positive semi-definite you can no longer guarantee that the distance between two points is positive. Besides the matrix should also be symmetric, therefore the distance from A to B should be the same as the distance from B to A.
I don’t know, but my guess is you won’t necessarily get a good answer from your SVM. I think not having a positive semi-definite kernel is like having a set that’s no longer closed, where you can end up with results outside the set and thus don’t make sense.
These are called kernel tricks. The objective is to tranform data from input space to a higher dimensional feature space. Transforming data into higher dimension space raises the possibility of data being linearly seperable via a hyperplane.

Now if the kernels don’t follow the required characteristics, it would imply that all the operations performed in the feature space are not longer valid which will result in your model being crap, literally crap. So, if the kernel does not fulfill the required properties, best is not to do the transformtion and deal with the data in the input space.

### 14. Lasso vs Ridge
Ridge regression and Lasso Regression are similar methods that try to reduce overfitting of our regression. The difference is in the methods that were being implemented. In ridge regression, we add the squared of magnitude of coefficient, as a penalty into the cost function (also called L2 penalty). For Lasso, we add absolute value of coefficient as a penalty to the cost function (also called L1 penalty). Lasso is better if we want to train a sparse model: this is because L1 regularization forces parameters to be shrunk to 0 - thereby helps with feature selection.

In either cases, it is helpful to normalize variables before running a regularized regression. This is because we want the regularization penalty to be applied evenly across all variables.

Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function. Here the highlighted part represents L2 regularization element.
Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.
The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.

Penalty terms: L1 regularization uses the sum of the absolute values of the weights, while L2 regularization uses the sum of the weights squared. Feature selection: L1 performs feature selection by reducing the coefficients of some predictors to 0, while L2 does not. Computational efficiency: L2 has an analytical solution, while L1 does not. Multicollinearity: L2 addresses multicollinearity by constraining the coefficient norm.


Penalty terms: L1 regularization uses the sum of the absolute values of the weights, while L2 regularization uses the sum of the weights squared. Feature selection: L1 performs feature selection by reducing the coefficients of some predictors to 0, while L2 does not. Computational efficiency: L2 has an analytical solution, while L1 does not. Multicollinearity: L2 addresses multicollinearity by constraining the coefficient norm.

### 15. Xgboost vs Random Forest (bagging)
In bagging, we have several base learners or decision trees which generated in parallel and form the base learners of bagging technique. However, in boosting, the trees are built sequentially such that each subsequent tree aims to reduce the errors of the previous tree. Each tree learns from its predecessors and updates the residual errors. Hence, the tree that grows next in the sequence will learn from an updated version of the residuals.

In contrast to bagging techniques like Random Forest, in which trees are grown to their maximum extent, boosting makes use of trees with fewer splits.

Boosting is based on weak learners (high bias, low variance). … Boosting reduces error mainly by reducing bias (and also to some extent variance, by aggregating the output from many models). On the other hand, Random Forest uses as you said fully grown decision trees (low bias, high variance).
