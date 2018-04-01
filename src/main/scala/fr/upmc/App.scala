package fr.upmc

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.RandomForest


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

    println("Bouguetof - Erraoui - Kassar")
    val dataf = sqlContext.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("loan.csv")

    var data = dataf


    println("HI !")
    println("data count : " + data.count())
    println("filter useless lines")

    var currentData = data.filter(col("loan_status").equalTo("Current"))

    data = data.filter(!col("loan_status").equalTo("Oct-2015"))
    data = data.filter(!col("loan_status").equalTo("Current"))
    data = data.filter(!col("loan_status").startsWith("Iss"))

    data = data.withColumn("loan_status", when(col("loan_status").startsWith("Late"), "Late").otherwise(col("loan_status")))
    data = data.withColumn("loan_status", when(col("loan_status").equalTo("Does not meet the credit policy. Status:Fully Paid"), "Fully Paid").otherwise(col("loan_status")))
    data = data.withColumn("loan_status", when(col("loan_status").equalTo("Does not meet the credit policy. Status:Charged Off"), "Charged Off").otherwise(col("loan_status")))
    data = data.withColumn("loan_status", when(col("loan_status").startsWith("Default"), "Late").otherwise(col("loan_status")))
    data = data.withColumn("loan_status", when(col("loan_status").startsWith("I"), "Late").otherwise(col("loan_status")))
    data = data.withColumn("loan_status", when(col("loan_status").startsWith("Cha"), "Late").otherwise(col("loan_status")))

    println("data count : " + data.count())

    data.groupBy("loan_status").count.show

    def renameColumns(df: DataFrame) = df.select("*").withColumnRenamed("earliest_cr_line", "earliest_credit_line")
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

    data = renameColumns(data)
    currentData = renameColumns(currentData)


    //recupération des données des variables textuelles
    def normalize(s: String) = if (s == null) "" else s.replaceAll(" ", "").toLowerCase

    def fillList(column: String): List[String] = {
      data.select(column).distinct().rdd.filter(_ != null).map(x => normalize(x.getString(0))).filter(!_.equals(""))
        .collect().toList
    }

    println("Start collecting categorical data ")
    val loanStatus = fillList(("loan_status"))
    val homeOwnerships = fillList(("home_ownership"))
    val grades = fillList(("grade"))
    // val titles = fillList(("title")) // 50000/200000 *********************
    val purposes = fillList(("purpose"))
    val emp_lengths = fillList(("emp_length"))
    //  val emp_titles = fillList(("emp_title")) // 150000/200000 ***********************
    val sub_grades = fillList(("sub_grade"))
    println("End collecting categorical data ")

    println("begin cleaning")


    def normalizeData(df: DataFrame) = df.select("loan_status", "dti", "delinquent_2_years", "inquiries_last_6_months",
      "zip_code", "home_ownership", "grade", "installment", "term", "loan_amount", "funded_amount",
      "investor_funds", "sub_grade", /*"emp_title",*/ "emp_length", "annual_income", "purpose", /*"title",*/
      "open_account", "public_records", "revol_bal", "revol_util", "total_acc")
      .toDF().rdd.filter(!_.toSeq.contains(null)).map(l => {
      try {
        val ss = l.toString().replaceAll("\\[", "").replaceAll("\\]", "").split(",");
        var i = -1

        def I = {
          i = i + 1;
          i
        }

        def convertS(s: String, l: List[String]) = {
          l.indexOf(normalize(s)) + 1
        }

        def convertLabel(s: String, l: List[String]) = {
          l.indexOf(normalize(s))
        }

        def convertD(s: String) = {
          s.toDouble
        }

        LabeledPoint(convertLabel(ss(I), loanStatus), Vectors.dense(
          convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I).substring(0, 3)), convertS(ss(I), homeOwnerships),
          convertS(ss(I), grades), convertD(ss(I)), convertD(ss(I).split("months")(0)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)),
          convertS(ss(I), sub_grades), /*convertS(ss(I), emp_titles),*/ convertS(ss(I), emp_lengths), convertD(ss(I)), convertS(ss(I), purposes),
          /*convertS(ss(I), titles),*/ convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I))
        ))
      } catch {
        case _ => null
      }
    }).filter(_ != null).cache()

    val cachedData = normalizeData(data)
    val cachedCurrentData = normalizeData(currentData)

    println("end cleaning\ndata count : " + cachedData.count())

    val splitedData = cachedData.randomSplit(Array(0.70, 0.30))
    val (trainingData, testData) = (splitedData(0), splitedData(1))

    println()
    // LogisticRegressionWithLBFGS
    println("Start LogisticRegressionWithLBFGS ...")
    val model1 = new LogisticRegressionWithLBFGS().setNumClasses(loanStatus.length).run(trainingData)
    var predictVsReal = testData.map(point => (model1.predict(point.features), point.label))

    var evaluation = new MulticlassMetrics(predictVsReal)
    println("Accuracy = " + evaluation.accuracy)

    println()
    val categoricalFeaturesInfo = Map(4 -> (homeOwnerships.length + 1), 5 -> (grades.length + 1), 12 /*13*/ -> (emp_lengths.length + 1))

    // DecisionTreeModel
    println("Start DecisionTreeModel ...")
    val model2: DecisionTreeModel = DecisionTree.trainClassifier(trainingData, numClasses = loanStatus.length, categoricalFeaturesInfo,
      impurity = "gini", maxDepth = 5, maxBins = (purposes.length + 1))

    predictVsReal = testData.map(point => (model2.predict(point.features), point.label))

    evaluation = new MulticlassMetrics(predictVsReal)
    println("Accuracy = " + evaluation.accuracy)

    println

    // RandomForest
    println("Start RandomForest ...")
    val model3 = RandomForest.trainClassifier(trainingData, numClasses = loanStatus.length, categoricalFeaturesInfo,
      numTrees = 5, featureSubsetStrategy = "auto", impurity = "gini", maxDepth = 5, maxBins = (purposes.length + 1))
    predictVsReal = testData.map(point => (model3.predict(point.features), point.label))

    evaluation = new MulticlassMetrics(predictVsReal)
    println("Accuracy = " + evaluation.accuracy)

    println
    println("Prediction for users with 'current' loan status using DecisionTree")
    val predictionForCurrentUsers = cachedCurrentData.map(point => model2.predict(point.features))
    val lateLabel = loanStatus.indexOf(normalize("Late"))
    val koCount = predictionForCurrentUsers.filter(_ == lateLabel).count()
    val okCount = predictionForCurrentUsers.count() - koCount

    println
    println("Number of members that will probably repay their loans = " + okCount + " / " + predictionForCurrentUsers.count())
    println("Number of members that will probably not repay their loans with respected delay = " + koCount + " / " + predictionForCurrentUsers.count())
  }
}
