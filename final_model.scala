/*
*	Proyecto de Clasificacion
*   Etapa 4: Modelo Final
*/


/*CONSTRUCCCION DEL MODELO FINAL*/
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 30
val maxBins = 100
val model_dt = DecisionTree.trainClassifier(labeledPoints, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

model_dt.save(sc, "/home/yanelis/Downloads/myDecisionTreeClassificationModel")


val modelType= "multinomial"
val modelNB = NaiveBayes.train(labeledPoints, 0.1, modelType)
modelNB.save(sc, "/home/yanelis/Downloads/myNaiveBayesClassificationModel")
