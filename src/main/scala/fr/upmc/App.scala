package fr.upmc


import java.io.FileWriter
import java.util
import java.util.stream.Collectors

import org.apache.spark._
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer

/**
  * Hello world!
  * Hello world!
  *
  */
object App {


  def main(args: Array[String]) {

    //val conf = new SparkConf()
    //conf.setMaster("local[2]").setAppName("Analyse Loan Data")


    //val sc = new SparkContext(conf)

    val sqlContext = SparkSession
      .builder()
      .appName("Analyse Loan Data").master("local")
      .getOrCreate()

    sqlContext.sparkContext.setLogLevel("ERROR")

    println("Hello World!")
    val dataf = sqlContext.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("loan.csv")

    println("HI !")

    var cleanDF = dataf.select("*").withColumnRenamed("earliest_cr_line", "earliest_credit_line")
      .withColumnRenamed("inq_last_6mths", "inquiries_last_6_months")
      .withColumnRenamed("int_rate", "interest_rate")
      .withColumnRenamed("loan_amnt", "loan_amount")
      .withColumnRenamed("funded_amnt", "funded_amount")
      .withColumnRenamed("funded_amnt_inv", "investor_funds")
      .withColumnRenamed("int_rate", "interest_rate")
      .withColumnRenamed("annual_inc", "annual_income")
      .withColumnRenamed("delinq_2yrs", "delinquent_2_years")
      .withColumnRenamed("mths_since_last_delinq", "months_since_last_delinquent")
      .withColumnRenamed("open_acc", "open_account")
      .withColumnRenamed("pub_rec", "public_records")
    
    //.withColumnRenamed("open_acc", "open_account").limit(100).show()
    // useless : issue_d - pymnt_plan - url - desc - addr_state
    // recheck : earliest_cr_line - mths_since_last_delin - mths_since_last_record - initial_list_status
    //println(cleanDF.select("loan_status").distinct().rdd.foreach(println))

    /*
    *dataf.withColumn("loan_status", when(col("loan_status").startsWith("Late"),"Late").otherwise(col("loan_status")))
     */

    println(cleanDF.count())



    //recupération des données des variables textuelles

    def normalize(s: String) = if (s == null) "" else s.replaceAll(" ", "").toLowerCase

    //    import  scala.collection.JavaConversions._
    def fillList(column: String): List[String] = {
      cleanDF.select(column).distinct().rdd.filter(_ != null).map(x => normalize(x.getString(0))).filter(!_.equals(""))
        .collect().toList
    }

    val loanStatus = fillList(("loan_status"))
    val homeOwnerships = fillList(("home_ownership"))
    val grades = fillList(("grade"))
    val titles = fillList(("title"))
    val purposes = fillList(("purpose"))
    val emp_lengths = fillList(("emp_length"))
    val emp_titles = fillList(("emp_title"))
    val sub_grades = fillList(("sub_grade"))

    println("begin cleanning")
    import sqlContext.implicits._
    val cachedData = cleanDF.select("loan_status", "dti", "delinquent_2_years", "inquiries_last_6_months",
      "zip_code", "home_ownership", "grade", "installment", "term", "loan_amount", "funded_amount",
      "investor_funds", "sub_grade", "emp_title", "emp_length", "annual_income", "purpose", "title",
      "open_account", "public_records", "revol_bal", "revol_util", "total_acc")
      .toDF().rdd.map(l => {
      val ss = l.toString().replaceAll("\\[", "").replaceAll("\\]", "").split(",");

      try {
        var i = -1

        def I = {
          i = i + 1;
          i
        }


        def convertS(s: String, l: List[String]) = {
          if (s == null) "unknown".hashCode else l.indexOf(normalize(s)) * 2
        }

        def convertLabel(s: String, l: List[String]) = {
          val temp = normalize(s)
          val i = l.indexOf(if (temp.startsWith("late")) "late" else temp)
          if (i < 0) throw new RuntimeException("problem") else i
        }

        def convertD(s: String) = {
          s.toDouble
        }

        LabeledPoint(convertLabel(ss(I), loanStatus), Vectors.dense(
          convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I).substring(0, 3)), convertS(ss(I), homeOwnerships),
          convertS(ss(I), grades), convertD(ss(I)), convertD(ss(I).split("months")(0)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)),
          convertS(ss(I), sub_grades), convertS(ss(I), emp_titles), convertS(ss(I), emp_lengths), convertD(ss(I)), convertS(ss(I), purposes),
          convertS(ss(I), titles), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I))
        ))
      } catch {
        case _ => null
      }
    }).filter(l => l != null).cache()
    println("end cleanning")

/*
    println("saving in a file")
    try {
      cachedData.map { case LabeledPoint(label, vector) =>
        var line: String = label.toString

        for (x <- vector.toArray) {
          line = line + "," + x.toString
        }
        line
      }.saveAsTextFile("path");
    } catch {
      case _ => println("saving error")
    }

    println("saving end")
*/
    println(cachedData.count())

    //    val cachedData = readData(sqlContext)
    println(cachedData.count())
    //cachedData.saveAsTextFile()
    // split du data set en données d'apprentissage et données de tests
    val splitedData = cachedData.randomSplit(Array(0.70, 0.30))
    val (trainingData, testData) = (splitedData(0), splitedData(1))

    // création du modèle d'aprentissage
    val model = new LogisticRegressionWithLBFGS().setNumClasses(loanStatus.length).run(trainingData)

    // test du modèle
    val preditVSreel = testData.map(point => (model.predict(point.features), point.label))

    // score
    val evaluation = new MulticlassMetrics(preditVSreel)

    println(evaluation.confusionMatrix)
    println(evaluation.recall)
    println(evaluation.precision)
    println(evaluation.fMeasure)
    /*
        869595
        0.0  185.0   0.0  0.0  1.0   0.0   0.0  3918.0    0.0
        0.0  9867.0  0.0  0.0  36.0  10.0  0.0  49802.0   14.0
        0.0  20.0    0.0  0.0  0.0   1.0   0.0  363.0     0.0
        0.0  84.0    0.0  0.0  0.0   0.0   0.0  1744.0    0.0
        0.0  307.0   0.0  0.0  82.0  0.0   0.0  120.0     51.0
        0.0  1799.0  0.0  0.0  18.0  15.0  0.0  11337.0   7.0
        0.0  50.0    0.0  0.0  0.0   0.0   0.0  2428.0    1.0
        0.0  5545.0  0.0  0.0  41.0  23.0  0.0  172114.0  2.0
        0.0  95.0    0.0  0.0  33.0  1.0   0.0  36.0      31.0
        0.6999319704359658
        0.6999319704359658
        0.6999319704359658
    */

    val fw = new FileWriter("test.txt", true)
    /*
        for (x <- 2 to 10) {
          println("---------------- k = " + x + "  -------------")
          val clusters = KMeans.train(cachedData, x, 20)
          clusters.clusterCenters.foreach(println)
          val WSSSE = clusters.computeCost(cachedData)
          println("---------------- WSSSE = " + WSSSE + "  -------------")
          try {
            fw.write("\nk = " + x + "\nWSSSE = " + WSSSE)
            fw.flush()
          }
        }
        */
    fw.close()

  }

  def readData(sqlContext: SparkSession) = sqlContext.sparkContext.textFile("path", 1).map(_.split(","))
    .map(line => LabeledPoint(line(0).toDouble, Vectors.dense(line.tail.map(_.toDouble))))
}
