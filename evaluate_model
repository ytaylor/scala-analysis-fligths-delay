/*
*	Proyecto de Clasificacion
*   Etapa 3:Evaluacion del modelo
*/


/*Evaluacion del modelo sobre el conjunto de prueba*/

/*DECISION TREE*/
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 30
val maxBins = 100

val model_dt = DecisionTree.trainClassifier(testData, numClasses, categoricalFeaturesInfo, impurity, 30, maxBins)
val predictionsAndLabels_dt= testData.map{d => (model_dt.predict(d.features), d.label)}
val binarymetrics_dt = new BinaryClassificationMetrics(predictionsAndLabels_dt)
val metrics_dt = new MulticlassMetrics(predictionsAndLabels_dt)

/*Tasa de error, su desviación estándar y su intervalo de confianza para una confianza del 95%.*/
val acierto_dt = metrics_dt.accuracy
val error_dt = 1 - acierto_dt

val errores_dt = predictionsAndLabels_dt.map(x => if (x._1 == x._2) 0 else 1).sum
val tasaError_dt= errores_dt/predictionsAndLabels_dt.count

val desvStandr_dt = Math.sqrt(error_dt*(tasaError_dt)/predictionsAndLabels_dt.count)
val intervaloconfianza_positive_dt= error_dt + 2.4729 * desvStandr_dt
val intervaloconfianza_negative_dt = error_dt - 2.4729 * desvStandr_dt

/*Tasa de ciertos y falsos positivos.*/
val labels = metrics_dt.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics_dt.precision(l), metrics_dt.falsePositiveRate(l), metrics_dt.truePositiveRate(l))
}

/*Precision*/
val precision_dt = binarymetrics_dt.precisionByThreshold()
precision_dt.foreach { 
  case (t, p) => {
    println(s"Threshold is: $t, Precision is: $p")
    if (t == 0.5) {
      println(s"Desired: Threshold is: $t, Precision is: $p")        
    } } }

/* Recall.*/

val recall_dt = binarymetrics_dt.recallByThreshold()
recall_dt.foreach { 
  case (t, p) => {
    println(s"Threshold is: $t, Recall is: $p")

    if (t == 0.5) {
      println(s"Desired: Threshold is: $t, Precision is: $p")        
    }}}

/* Área bajo la curva ROC*/
val curvaROC_dt =binarymetrics_dt.roc
val areaUnderROC_dt = binarymetrics_dt.areaUnderROC
println("DT auROC: " + areaUnderROC_dt.toString)
curvaROC_dt.take(10)

/*Área bajo la curva PR.*/
val tree_areaUnderPR = binarymetrics_dt.areaUnderPR



/**********************************************************************************************/

/*NAIVE BAYES*/
val modelType= "multinomial"
val model_nb = NaiveBayes.train(testData, 0.1, modelType)
val predictionsAndLabels_nb = testData.map{d => (model_nb.predict(d.features), d.label)}

val binarymetrics_nb = new BinaryClassificationMetrics(predictionsAndLabels_nb)
val metrics_nb = new MulticlassMetrics(predictionsAndLabels_nb)

/*Tasa de error, su desviación estándar y su intervalo de confianza para una confianza del 95%.*/
val errores_nb = predictionsAndLabels_nb.map(x => if (x._1 == x._2) 0 else 1).sum
val tasaError_nb= errores_nb/predictionsAndLabels_nb.count
val desvStandr_nb = Math.sqrt(tasaError_nb*(1-tasaError_nb)/predictionsAndLabels_nb.count)

val intervaloconfianza_positive_nb= tasaError_nb + 2.4729 * desvStandr_nb
val intervaloconfianza_negative_nb= tasaError_nb - 2.4729 * desvStandr_nb


/*Tasa de ciertos y falsos positivos.*/
val labels = metrics_nb.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics_nb.precision(l), metrics_nb.falsePositiveRate(l), metrics_nb.truePositiveRate(l))
}

/*Precision*/
val precision_nb = binarymetrics_nb.precisionByThreshold()
precision_nb.foreach { 
  case (t, p) => {
    println(s"Threshold is: $t, Precision is: $p")

    if (t == 0.5) {
      println(s"Desired: Threshold is: $t, Precision is: $p")        
    }}}


/* Recall.*/
val recall_nb = binarymetrics_nb.recallByThreshold()
recall_nb.foreach { 
  case (t, p) => {
    println(s"Threshold is: $t, Recall is: $p")
    if (t == 0.5) {
      println(s"Desired: Threshold is: $t, Precision is: $p")        
    }}}

/* Área bajo la curva ROC*/
val curvaROC_nb =binarymetrics_nb.roc
val auROC_nb= binarymetrics_nb.areaUnderROC

println("NB auROC: " + auROC_nb.toString)


/*Área bajo la curva PR.*/
val areaUnderPR_nb= binarymetrics_nb.areaUnderPR

