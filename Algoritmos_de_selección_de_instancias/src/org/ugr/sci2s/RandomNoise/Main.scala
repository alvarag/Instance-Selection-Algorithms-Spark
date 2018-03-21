package org.ugr.sci2s.RandomNoise

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.util.Random

object Main extends Serializable {

  def main(args: Array[String]) {

    val input = args(0)
    val output = args(1)
    val pct = args(2).toDouble

    val conf = new SparkConf().setAppName("RandomNoise: " + pct)
    /*.setMaster("local[*]")
      .setExecutorEnv("spark.driver.memory", "10g")
      .setExecutorEnv("spark.executor.memory", "10g")
      .setExecutorEnv("spark.executor.instances", "6")*/
    val sc = new SparkContext(conf)

    val rawDataTrain = sc.textFile(input)
    val trainingData = rawDataTrain.map { line =>
      val values = line.split(',').map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last
      LabeledPoint(label, featureVector)
    }.cache()

    val labels = trainingData.map(_.label).distinct().collect()

    val tam = trainingData.count.toInt

    val num = math.round(tam * (pct.toDouble / 100))

    val range = util.Random.shuffle(0 to tam - 1)

    val indices = range.take(num.toInt)

    val broadcastInd = sc.broadcast(indices)

    val noiseData = trainingData.zipWithIndex.map {
      case (v, k) =>
        if (broadcastInd.value contains (k)) {
          val label = labels diff List(v.label)
          LabeledPoint(label(Random.nextInt(label.length)), v.features)
        } else {
          v
        }
    }
    noiseData.map(l => l.features.toArray.mkString(",") + "," + l.label).repartition(1).saveAsTextFile(output)
  }
}
