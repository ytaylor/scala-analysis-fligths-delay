import org.apache.spark.rdd.RDD
import org.apache.spark.mllib

val DATA = PATH + FILE

val lines = sc.textFile(DATA)
val nonEmpty = lines.filter(_.nonEmpty)
val parsed = nonEmpty.map(line => line.split(","))

parsed.first
