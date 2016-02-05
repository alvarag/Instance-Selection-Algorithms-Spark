package instanceSelection.demoIS

import java.util.logging.Level
import java.util.logging.Logger

import scala.collection.mutable.MutableList

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import classification.seq.knn.KNN
import instanceSelection.abstr.TraitIS
import instanceSelection.seq.abstr.TraitSeqIS
import instanceSelection.seq.cnn.CNN
import utils.partitioner.RandomPartitioner

/**
 * Algoritmo de selección de instancias Democratic Instance Selection.
 *
 * Este algoritmo se basa en dividir el conjunto original en múltiples
 * subconjuntos y aplicar sobre cada uno de ellos un algoritmo de selección
 * de instancias más simple. Con la salida de este proceso, repetido durante
 * un número indicado de veces, calcularemos la verdadera salida del algoritmo
 * en base a un criterio que busca adecuar precisión y tamaño del conjunto
 * final.
 *
 * Participante en el patrón de diseño "Strategy" en el que actúa con el
 * rol de estrategia concreta ("concrete strategies"). Hereda de la clase que
 * participa como estrategia ("Strategy")
 * [[instanceSelection.abstr.TraitIS]].
 *
 * @constructor Crea un nuevo selector de instancias con los parámetros por
 *   defecto.
 *
 * @author  Alejandro González Rogel
 * @version 1.0.0
 */
class DemoIS extends TraitIS {

  /**
   * Ruta al fichero que guarda los mensajes de log.
   */
  private val bundleName = "resources.loggerStrings.stringsDemoIS"
  /**
   * Logger.
   */
  private val logger = Logger.getLogger(this.getClass.getName(), bundleName)

  /**
   * Número de repeticiones que se realizará la votación.
   */
  var numRepeticiones = 10

  /**
   *  Peso de la precisión sobre el tamaño al calcular el criterio de selección.
   */
  var alpha = 0.75

  /**
   *  Número de paticiones en las que dividiremos el conjunto de datos para
   *  las realizar las votaciones.
   */
  var numPartitions = 10

  /**
   * Semilla para el generador de números aleatorios.
   */
  var seed: Long = 1

  /**
   * Número de vecinos cercanos a utilizar en el KNN.
   */
  var k = 1

  /**
   * Porcentaje del conjunto de datos inicial usado para calcular la precisión
   * de las diferentes aproximaciones según el número de votos de las instancias.
   */
  var datasetPerc = 1.0

  override def instSelection(
    sc: SparkContext,
    parsedData: RDD[LabeledPoint]): RDD[LabeledPoint] = {

    // Añadimos a las instancias un contador para llevar la cuenta del número
    // de veces que han sido seleccionadas en el siguiente paso
    val RDDconContador = parsedData.map(instance => (0, instance))

    val resultIter = doIterations(RDDconContador)

    val (indexBestCrit, bestCrit) = lookForBestCriterion(resultIter)

    resultIter.filter(
      tupla => tupla._1 < indexBestCrit).map(tupla => tupla._2)

  }

  /**
   * Durante un número fijado de veces, reparticionará el conjunto original
   * y aplicará sobre cada subconjunto un algoritmo de selección de instancias,
   * actualizando los votos de aquellas instancias que han sido no seleccionadas.
   *
   * @param RDDconContador Conjunto de datos original y donde cada instancia
   *   tiene un contador asociado que indica el número de veces que ha sido
   *   seleccionada.
   * @return Conjunto de datos original con los contadores actualizados.
   *
   * @todo Habilitar la posibilidad de cambiar el algoritmo de selección
   * de instancias usado cuando exista más de uno implementado.
   */
  private def doIterations(
    RDDconContador: RDD[(Int, LabeledPoint)]): RDD[(Int, LabeledPoint)] = {

    // Algoritmo secuencial a usar en cada subconjunto de datos.
    val seqAlgorithm: TraitSeqIS = new CNN()

    // Operación a realizar en cada uno de los nodos.
    val votingInNodes = new VotingInNodes()
    // RDD usada en la iteración concreta.
    var actualRDD = RDDconContador
    // Particionador para asignar nuevas particiones a las instancias.
    val partitioner = new RandomPartitioner(numPartitions, seed)

    for { i <- 0 until numRepeticiones } {
      // Redistribuimos las instancias en los nodos
      actualRDD = actualRDD.partitionBy(
        partitioner)

      // En cada nodo aplicamos el algoritmo de selección de instancias
      actualRDD = actualRDD.mapPartitions(instancesIterator =>
        votingInNodes
          .applyIterationPerPartition(instancesIterator, seqAlgorithm))
    }
    actualRDD.persist()
    actualRDD.name = "InstancesAndVotes"
    actualRDD
  }

  /**
   * Calcula la adecuación de cada una de las posibles soluciones y encuentra
   * aquella cuyo valor sea más bajo.
   *
   * @param  RDDconContador Conjunto de datos original. Cada instancia
   *   tiene un contador asociado que indica el número de veces que ha sido
   *   seleccionada.
   *
   * @return Tupla con el número máximo de veces que la instancia no ha
   *   sido seleccionada y que conduce al criterio más bajo y el propio valor de
   *   dicho criterio.
   *
   */
  private def lookForBestCriterion(
    RDDconContador: RDD[(Int, LabeledPoint)]): (Int, Double) = {

    // Donde almacenar los valores del criterion de cada número de votos.
    var criterion = new MutableList[Double]
    // Creamos un conjunto de test
    val testRDD =
      RDDconContador.sample(false, datasetPerc / 100, seed).map(tupla => tupla._2)
        .collect()

    val originalDatasetSize = RDDconContador.count()

    for { i <- 1 to numRepeticiones } {

      // Seleccionamos todas las instancias que no superan el tresshold parcial
      val selectedInst = RDDconContador.filter(
        tupla => tupla._1 < i).map(tupla => tupla._2).collect()

      if (selectedInst.isEmpty) {
        criterion += Double.MaxValue
      } else {
        criterion += calcCriterion(selectedInst, testRDD,
          originalDatasetSize)
      }
    }

    (criterion.indexOf(criterion.min) + 1, criterion.min)
  }

  /**
   *
   * @param  selectedInst  Subconjunto de instancias cuyo contador ha superado
   *   un número determinado
   * @param  dataSetSize  Tamaño del conjunto original de datos.
   * @return Resultado numérico del criterio de selección.
   */
  private def calcCriterion(selectedInst: Array[LabeledPoint],
                            testRDD: Array[LabeledPoint],
                            dataSetSize: Long): Double = {

    // Calculamos el porcentaje del tamaño que corresponde al subconjunto
    val subDatasize = selectedInst.size.toDouble / dataSetSize

    // Calculamos la tasa de error
    val knn = new KNN()
    val knnParameters: Array[String] = Array.ofDim(2)
    knnParameters(0) = "-k"
    knnParameters(1) = k.toString()
    knn.setParameters(knnParameters)
    knn.train(selectedInst)
    var failures = 0
    for { instancia <- testRDD } {
      var result = knn.classify(instancia)
      if (result != instancia.label) {
        failures += 1
      }
    }
    val testError = failures.toDouble / testRDD.size

    // Calculamos el criterio de selección.
    alpha * testError + (1 - alpha) * subDatasize
  }

  override def setParameters(args: Array[String]): Unit = {

    // Comprobamos si tenemos el número de atributos correcto.
    checkIfCorrectNumber(args.size)

    for { i <- 0 until args.size by 2 } {
      try {
        val identifier = args(i)
        val value = args(i + 1)
        assignValToParam(identifier, value)
      } catch {
        case ex: NumberFormatException =>
          logger.log(Level.SEVERE, "DemoISNoNumberError", args(i + 1))
          throw new IllegalArgumentException()
      }
    }

    if (numRepeticiones <= 0 || alpha < 0 || alpha > 1 || k <= 0 ||
      numPartitions <= 0 || datasetPerc <= 0 || datasetPerc >= 100) {
      logger.log(Level.SEVERE, "DemoISWrongArgsValuesError")
      logger.log(Level.SEVERE, "DemoISPossibleArgs")
      throw new IllegalArgumentException()
    }

  }

  protected override def assignValToParam(identifier: String,
                                          value: String): Unit = {
    identifier match {
      case "-rep"    => numRepeticiones = value.toInt
      case "-alpha"  => alpha = value.toDouble
      case "-s"      => seed = value.toInt
      case "-k"      => k = value.toInt
      case "-np"     => numPartitions = value.toInt
      case "-dsperc" => datasetPerc = value.toDouble
      case somethingElse: Any =>
        logger.log(Level.SEVERE, "DemoISWrongArgsError",
          somethingElse.toString())
        logger.log(Level.SEVERE, "DemoISPossibleArgs")
        throw new IllegalArgumentException()
    }
  }

  /**
   * Comprueba si el número de parámetros introducido puede corresponder a
   * al de una configuración válida.
   *
   * @param Número de parametros y valores introducidos al configurar.
   *
   * @throws IllegalArgumentException si el número de parametros no es
   *   correcto
   */
  @throws(classOf[IllegalArgumentException])
  def checkIfCorrectNumber(argsNumber: Int): Unit = {

    if (argsNumber % 2 != 0) {
      logger.log(Level.SEVERE, "DemoISPairNumberParamError",
        this.getClass.getName)
      throw new IllegalArgumentException()
    }

  }

}
