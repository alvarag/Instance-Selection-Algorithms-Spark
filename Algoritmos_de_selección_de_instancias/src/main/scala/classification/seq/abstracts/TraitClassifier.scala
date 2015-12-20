package classification.seq.abstracts

import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Interfaz que define los métodos bñasicos que deberá contener un clasificador.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
abstract class TraitClassifier {

  /**
   * Realiza la clasificación de una determinada instancia.
   *
   * @param inst  Instancia a clasificar.
   * @retun Número correspondiente a la clase predicha por el clasificador.
   */
  def classify(inst: LabeledPoint): Double
  
    /**
   * Realiza la clasificación de un conjunto de instancias
   *
   * @param inst  Instancias a clasificar.
   * @retun Número correspondiente a la clase predicha por el clasificador.
   */
  def classify(inst: Iterable[LabeledPoint]): Array[Double]

}
