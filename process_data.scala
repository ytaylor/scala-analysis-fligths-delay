/*
*	Proyecto de Clasificacion
*   Etapa 1: Procesamiento de Datos
*/

/* Importacion de las librerias necesarias*/
import org.apache.spark.mllib
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

/* Lectura del conjunto de datos*/
val PATH ="/home/floppita/Downloads/"
val FILE= "/vuelos.data"
val DATA = PATH + FILE

val lines = sc.textFile(DATA)

/*Hay que hacer un proceso de filtrado pues no todos los datos tienen valores en limpio*/

val nonEmpty = lines.filter(_.nonEmpty)
val parsed = nonEmpty.map(line => line.split(","))

parsed.first


/*Convertir losa atributos categoricos en Int*/
/*Creamos una colección Map, que nos facilitará transformar las etiquetas en Int ):*/

val day_month = parsed.map{a => a(0)}.distinct.collect
val day_week = parsed.map{a => a(1)}.distinct.collect
val carrier_code = parsed.map{a => a(2)}.distinct.collect
val tail_num = parsed.map{a => a(3)}.distinct.collect
val flnum = parsed.map{a => a(4)}.distinct.collect
val orig_id = parsed.map{a => a(5)}.distinct.collect
val airport_orig = parsed.map{a => a(6)}.distinct.collect
val dest_id = parsed.map{a => a(7)}.distinct.collect
val airport_dst = parsed.map{a => a(8)}.distinct.collect
val crs_deptime = parsed.map{a => a(9)}.distinct.collect
val diez = parsed.map{a => a(10)}.distinct.collect
val once = parsed.map{a => a(11)}.distinct.collect
val delay_departure  = parsed.map{a => a(12)}.distinct.collect
val trece = parsed.map{a => a(13)}.distinct.collect
val catorce = parsed.map{a => a(14)}.distinct.collect
val quince = parsed.map{a => a(15)}.distinct.collect
val dist= parsed.map{a => a(16)}.distinct.collect
/* zipWithIndex, aplicado a una colección crea otra colección de pares
(elemento, índice)*/
val day_monthToNumeric = day_month.zipWithIndex.toMap
val day_weekToNumeric = day_week.zipWithIndex.toMap
val carrier_codeToNumeric = carrier_code.zipWithIndex.toMap
val tail_numToNumeric = tail_num.zipWithIndex.toMap
val orig_idToNumeric = orig_id.zipWithIndex.toMap
val onceToNumeric = once.zipWithIndex.toMap
val flnumToNumeric = flnum.zipWithIndex.toMap
val airport_origToNumeric = airport_orig.zipWithIndex.toMap
val dest_idToNumeric = dest_id.zipWithIndex.toMap
val airport_dstToNumeric = airport_dst.zipWithIndex.toMap
val crs_deptimeToNumeric = crs_deptime.zipWithIndex.toMap
val diezToNumeric = diez.zipWithIndex.toMap
val delay_departureToNumeric = delay_departure.zipWithIndex.toMap
val treceToNumeric = trece.zipWithIndex.toMap
val catorceToNumeric = catorce.zipWithIndex.toMap
val quinceToNumeric = quince.zipWithIndex.toMap
val distToNumeric = dist.zipWithIndex.toMap


/*Funcion para la creacion de la clase*/
/*Verifica si el vuelo esra retrasado, tiene mas de 30 minutos de retraso*/
def parseData(vals: String): Double = { if (vals.toDouble>=40) 1.0 else 0.0 }

/* Creacion del LabeledPoint*/
/*LabeledPoint representa una instancia etiquetada; contiene la clase
y las características
La etiqueta se almacena como un Double
Las características como un Vector
*/
val labeledPoints = parsed.map{ a => LabeledPoint(parseData(a(13)), Vectors.dense(day_monthToNumeric(a(0)), day_weekToNumeric (a(1)), carrier_codeToNumeric (a(2)), tail_numToNumeric(a(3)), flnumToNumeric(a(4)), orig_idToNumeric(a(5)), airport_origToNumeric(a(6)), dest_idToNumeric(a(7)), airport_dstToNumeric(a(8)), crs_deptimeToNumeric(a(9)), diezToNumeric(a(10)), onceToNumeric(a(11)), delay_departureToNumeric(a(12)), treceToNumeric(a(13)), catorceToNumeric(a(14)), quinceToNumeric(a(15)), distToNumeric(a(16))))}



/* Separacion del conjunto de datos, en conjunto de entrenamiento y de prueba*/
val dataSplits = labeledPoints.randomSplit(Array(0.80, 0.20)) 
val trainingData = dataSplits(0) 
val testData = dataSplits(1)

testData.saveAsTextFile(PATH + "testData.data")

trainingData.collect().foreach(println)
testData.collect().foreach(println)

trainingData.cache
testData.cache
