package classification.seq.knn

import java.util.logging.Level
import java.util.logging.Logger

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.MutableList

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

import classification.seq.abstr.TraitSeqClassifier
import utils.DistCalculator

/**
 * Class used by LSHKNN for computing the exhaustive nearest neighbours of an
 * instance on a given data set.
 *
 * @constructor A new exhaustive kNN search.
 *
 * @author Álvar Arnaiz-González
 * @version 2.0
 */
class KNNSequential(k: Int = 1) extends Serializable {

  /**
   * Distance calculator.
   */
  val distCalc = new DistCalculator


  /**
   * Distance between an instance and a collection of centroids.
   * 
   * @param Tuple composed of (instance, indx) and an array of centroids.
   * @return (Indx, Instance, Distances)
   */
  def distance(tuple: Tuple2[(Long, Vector), Array[Vector]]): (Long, Vector, List[Double]) = {
    var indx = tuple._1._1
    var instance = tuple._1._2
    var centroids = tuple._2
    
    // Compute the distance between the instance and the centroids.
    var distances = for { actualInst <- centroids }
      yield (euclideanDistance(instance.toArray, actualInst.toArray))

    (indx, instance, distances.toList)
  }
  
  /**
   * Distance between an instance and a collection of centroids.
   * 
   * @param Tuple composed of instance and an array of centroids.
   * @return (Instance, Distances)
   */
  def distance(tuple: Tuple2[LabeledPoint, Array[Vector]]): (LabeledPoint, List[Double]) = {
    var instance = tuple._1
    var centroids = tuple._2
    
    // Compute the distance between the instance and the centroids.
    var distances = for { actualInst <- centroids }
      yield (euclideanDistance(instance.features.toArray, actualInst.toArray))

    (instance, distances.toList)
  }
  
  
  def euclideanDistance(point1: Array[Double],
                        point2: Array[Double]): Double = {

    distCalc.euclideanDistance(point1, point2)
  }

  /**
   * Function for classifying instances.
   *
   * @param A tuple, first element: tuple (inst_id, inst), second element a
   *        list of instances for training.
   * @return A tuple, first element: inst_id, second element: predicted_class.
   */
  def mapClassify(tuple: ((Long, Vector), Iterable[LabeledPoint])): (Long, Double) =
    (tuple._1._1, classify(tuple._2, tuple._1._2))

  /**
   * Performs an exhaustive kNN search on a given data set.
   * If the training data set is empty, zero is returned.
   *
   * @param trainingData Training data set.
   * @param inst Instance to query.
   */
  def classify(trainingData: Iterable[LabeledPoint], inst: Vector): Double = {
    // Return 0 if the training data set is empty.
    if (trainingData.size == 0) {
      println("Not NN instances found")
      0.0
    } else {
      val closest = kNearestClasses(trainingData, inst)

      // Compute the most probable class and return it.
      val classification = closest.groupBy(t => t._1).maxBy(t => t._2.length)

      classification._1
    }
  }

  /**
   * Performs an exhaustive kNN search on a given data set.
   *
   * @param trainingData Training data set.
   * @param inst Instance to query.
   */
  def kNearestClasses(trainingData: Iterable[LabeledPoint], inst: Vector): ListBuffer[(Double, Double)] = {
    // Compute the distance between inst and training instances.
    val distances = for { actualInst <- trainingData }
      yield (actualInst.label, distCalc.euclideanDistance(
      inst.toArray, actualInst.features.toArray))

    var closest: ListBuffer[(Double, Double)] = ListBuffer.empty
    var iter = distances.iterator
    closest += iter.next()

    // Find out the nearest neighbours.
    while (iter.hasNext) {
      var actualInst = iter.next
      // If there aren't all kNN stored
      if (closest.size < k) {
        closest += actualInst
      } else {
        var maxDist = closest.maxBy((t) => t._2)._2
        // If the distance to an instance is smaller than the previously stored
        if (actualInst._2 < maxDist) {
          closest(closest.indexOf(closest.maxBy((t) => t._2))) = actualInst
        }
      }
    }

    closest.sortBy(tupla => tupla._2)
  }

}
