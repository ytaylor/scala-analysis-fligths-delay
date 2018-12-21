# scala-analysis-fligths-delay

This project use vuelos dataset fo perform predictive models using supervised learning algorithms and validate them cross way,  using Spark MLLib. This projetc include
* Definition of some Scala functions for preprocessing.
* Modeled with Spark and MLLib using Decision Trees and Naive Bayes.
* Build an interactive and predictive model for flight delays

## Preparation of data.
For the preparation of the data, we carry out cleaning, transformation, and main feature selection. For the analyzed data set the class does not finds explicit, so we have used our own function parseData () that is responsible for creating the delay class, we consider the flight delays of 40 minutes or more like delays and we mark it with a label of 1.0 and less than 40 minutes as not delay and we mark it with a label of 0.0.

We use Spark's RDDs to perform the same previous processing, transforming the raw flight delay data set in two feature matrices 
trainingData the training set and testData the test set. Further, we use the RDD cache method to ensure that these calculated RDDs (trainingData, testData) are cached in memory by Spark and not recalculated with each iteration.

With the data set trainingData (which we will use for training) and the set of data testData (which we will use for validation) as RDD, now we will build a predictive model using the Spark MLLib learning library.

## Creation and selection of models.
With basic characteristics of the data set and created the RDD, we proceed to the creation of the models. In order to compare the performance and the use of different models, the model is trained using Naïve Bayes and Decision Tree. 

### Cross validation

Cross validation is a critical part of the real world learning machine and it is fundamental for the selection of many models and the adjustment of parameters. The idea general behind the cross validation is that we want to know how our model is
will perform on unseen data. Cross-validation provides a mechanism in which we use part of our data set available to train our model and another part to evaluate the performance of this model.

As the model is tested with data that you have not seen during the training phase, when it is evaluated in this part of the data set, it gives us how well our model

It is generalized for the new data points.To carry out the cross-validation we have helped the MLUTils library of
MLIB, which has a kFold function that given the data set and number of iterations k performs partitions to the data set provided in sets of training and tests.
