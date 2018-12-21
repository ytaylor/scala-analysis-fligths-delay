/*
*	Proyecto de Clasificacion
*   Etapa 2: Creacion y seleccion del modelo
*/


/*ARBOLES DE DECISION*/ 

/*Buscando la profundidad del arbol con validacion cruzada interna*/

/*Creando los conjuntos de entreanmiento y prueba para la validacion cruzada, 
  Se ha utilizado la funcion kFold de la biblioteca MLUtils que dodo el numero
  de iteracciones k divide el conjunto en partes proporcionales que le permitan evaluar y entrenar*/
val cvdat = MLUtils.kFold(trainingData, 10, 42)


/* Funcion que entrena y evalua cada nivel de profundidad del arbol en los conjunto de pruebas e iteracciones. */
val evaluations = cvdat.map { case (train, test) => {
for(depth<-Array(10 ,20, 30)) 
yield {
val model=DecisionTree.trainClassifier(train, 7, Map[Int,Int](), "gini", depth, 100)
val predictionsAndLabels=test.map(example=>(model.predict(example.features), example.label))
val metrics = new MulticlassMetrics(predictionsAndLabels)
val binarymetrics = new BinaryClassificationMetrics(predictionsAndLabels)
(metrics.accuracy, binarymetrics.areaUnderROC, (depth, "gini", 100))
}}}


/*Imprimiendo la tasa de acierto por cada nivel del profundidad evaluado por cada iteraccion */
evaluations.map(a=>a.sortBy(_._2).reverse.foreach(println))

/*Calculando la tasa de error promedio*/
val avgError = evaluations.map(a=> a.map(_._1).reduce(_ + _) / evaluations.length)
val errormedia = avgError.sum / avgError.length


/*NAIVE BAYES*/
/* Buscando el valor de lambda para Naive Bayes con validacion cruzada interna*/
val evaluations_nb = cvdat.map { case (train, test) => {
for(lambda<-Array(0.1, 0.5, 0.20, 0.90)) 
yield {
val model=NaiveBayes.train(train, lambda, "multinomial")
val predictionsAndLabels=test.map(example=>(model.predict(example.features), example.label))
val metrics = new MulticlassMetrics(predictionsAndLabels)
val binarymetrics = new BinaryClassificationMetrics(predictionsAndLabels)
(metrics.accuracy, binarymetrics.areaUnderROC, lambda)
}}}

/*Imprimiendo la tasa de acierto por cada valor de lambda evaluado por cada iteraccion */
evaluations.map(a=> a.sortBy(_._2).reverse.foreach(println))

/*Calculando la tasa de error promedio*/
val avgError_nb = evaluations_nb.map(a=> a.map(_._1).reduce(_ + _) / evaluations_nb.length)
val errormedia_nb = avgError_nb.sum / avgError_nb.length
