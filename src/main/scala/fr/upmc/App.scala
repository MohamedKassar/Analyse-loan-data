package fr.upmc

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD


object App {


  def main(args: Array[String]) {

    if(args.length != 1){
      println("csv file path should be put in args")
      System.exit(1)
    }
    val sqlContext = SparkSession
      .builder()
      .appName("Analyse Loan Data").master("local")
      .getOrCreate()

    sqlContext.sparkContext.setLogLevel("ERROR")

    println("Bouguetof - Erraoui - Kassar")
    var data = sqlContext.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(args(0))


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

    def normalize(s: String) = if (s == null) "" else s.replaceAll(" ", "").toLowerCase

    def fillList(column: String, d: DataFrame): List[String] = {
      d.select(column).distinct().rdd.filter(_ != null).map(x => normalize(x.getString(0))).filter(!_.equals(""))
        .collect().toList
    }

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
    println("Start collecting categorical data ")
    val loanStatus = fillList("loan_status", data)
    val homeOwnerships = fillList("home_ownership", data)
    val grades = fillList("grade", data)
    val purposes = fillList("purpose", data)
    val emp_lengths = fillList("emp_length", data)
    val sub_grades = fillList("sub_grade", data)
    println("End collecting categorical data ")


    def normalizeData_20C(df: DataFrame) = df.select("loan_status", "dti", "delinquent_2_years", "inquiries_last_6_months",
      "zip_code", "home_ownership", "grade", "installment", "term", "loan_amount", "funded_amount",
      "investor_funds", "sub_grade", "emp_length", "annual_income", "purpose", "open_account", "public_records",
      "revol_bal", "revol_util", "total_acc")
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

    def normalizeData_NumericC(df: DataFrame) = df.select("loan_status", "dti", "delinquent_2_years", "inquiries_last_6_months",
      "installment", "loan_amount", "funded_amount",
      "investor_funds", "annual_income",
      "open_account", "public_records", "revol_bal", "revol_util", "total_acc")
      .toDF().rdd.filter(!_.toSeq.contains(null)).map(l => {
      try {
        val ss = l.toString().replaceAll("\\[", "").replaceAll("\\]", "").split(",");
        var i = -1

        def I = {
          i = i + 1;
          i
        }

        def convertLabel(s: String, l: List[String]) = {
          l.indexOf(normalize(s))
        }

        def convertD(s: String) = {
          s.toDouble
        }

        LabeledPoint(convertLabel(ss(I), loanStatus), Vectors.dense(
          convertD(ss(I)), convertD(ss(I)), convertD(ss(I)),
          convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)),
          convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I)), convertD(ss(I))
        ))
      } catch {
        case _ => null
      }
    }).filter(_ != null).cache()

    def normalizeData_4C(df: DataFrame) = df.select("loan_status", "dti", "loan_amount", "funded_amount", "annual_income")
      .toDF().rdd.filter(!_.toSeq.contains(null)).map(l => {
      try {
        val ss = l.toString().replaceAll("\\[", "").replaceAll("\\]", "").split(",");
        var i = -1

        def I = {
          i = i + 1;
          i
        }

        def convertLabel(s: String, l: List[String]) = {
          l.indexOf(normalize(s))
        }

        def convertD(s: String) = {
          s.toDouble
        }

        LabeledPoint(convertLabel(ss(I), loanStatus), Vectors.dense(
          convertD(ss(I)), convertD(ss(I)), convertD(ss(I)),
          convertD(ss(I))
        ))
      } catch {
        case _ => null
      }
    }).filter(_ != null).cache()


    println(" ------------------------  begin cleaning : only numeric features ----------------------------- ----------------------------------- ----------------------")
    val cachedDataNC = normalizeData_NumericC(data)
    println("end cleaning\ndata count : " + cachedDataNC.count())

    println("")
    val splitedDataNC = cachedDataNC.randomSplit(Array(0.70, 0.30), seed = 25896)
    val trainingDataNC = splitedDataNC(0)
    val testDataNC = splitedDataNC(1)

    println()
    // LogisticRegressionWithLBFGS
    println("Start LogisticRegressionWithLBFGS ...")
    val model1NC = new LogisticRegressionWithLBFGS().setNumClasses(loanStatus.length).run(trainingDataNC)
    val predictVsRealNC = testDataNC.map(point => (model1NC.predict(point.features), point.label))
    metrics(predictVsRealNC)

    println()
    var categoricalFeaturesInfo = Map[Int, Int]()

    // DecisionTreeModel
    println("Start DecisionTreeModel ...")
    val model2NC = DecisionTree.trainClassifier(trainingDataNC, numClasses = loanStatus.length, categoricalFeaturesInfo,
      impurity = "gini", maxDepth = 5, maxBins = 32)

    val predictVsRealNC1 = testDataNC.map(point => (model2NC.predict(point.features), point.label))
    metrics(predictVsRealNC1)
    println

    // RandomForest
    println("Start RandomForest ...")
    val model3NC = RandomForest.trainClassifier(trainingDataNC, numClasses = loanStatus.length, categoricalFeaturesInfo,
      numTrees = 5, featureSubsetStrategy = "auto", impurity = "gini", maxDepth = 5, maxBins = 32)
    val predictVsRealNC2 = testDataNC.map(point => (model3NC.predict(point.features), point.label))
    metrics(predictVsRealNC2)

    println

    println(" ------------------------  begin cleaning : numeric and textual features ----------------------------- ----------------------------------- ----------------------")

    val cachedDataNT = normalizeData_20C(data)
    val cachedCurrentData = normalizeData_20C(currentData)

    println("end cleaning\ndata count : " + cachedDataNT.count())

    println("")
    val splitedDataNT = cachedDataNT.randomSplit(Array(0.70, 0.30), seed = 25896)
    val trainingDataNT = splitedDataNT(0)
    val testDataNT = splitedDataNT(1)

    println()
    // LogisticRegressionWithLBFGS
    println("Start LogisticRegressionWithLBFGS ...")
    val model1NT = new LogisticRegressionWithLBFGS().setNumClasses(loanStatus.length).run(trainingDataNT)
    val predictVsRealNT = testDataNT.map(point => (model1NT.predict(point.features), point.label))
    metrics(predictVsRealNT)

    println()
    categoricalFeaturesInfo = Map(4 -> (homeOwnerships.length + 1), 5 -> (grades.length + 1), 12 /*13*/ -> (emp_lengths.length + 1))

    // DecisionTreeModel
    println("Start DecisionTreeModel ...")
    val model2NT = DecisionTree.trainClassifier(trainingDataNT, numClasses = loanStatus.length, categoricalFeaturesInfo,
      impurity = "gini", maxDepth = 5, maxBins = (purposes.length + 1))

    val predictVsRealNT1 = testDataNT.map(point => (model2NT.predict(point.features), point.label))
    metrics(predictVsRealNT1)
    println

    // RandomForest
    println("Start RandomForest ...")
    val model3NT = RandomForest.trainClassifier(trainingDataNT, numClasses = loanStatus.length, categoricalFeaturesInfo,
      numTrees = 5, featureSubsetStrategy = "auto", impurity = "gini", maxDepth = 5, maxBins = (purposes.length + 1))
    val predictVsRealNT2 = testDataNT.map(point => (model3NT.predict(point.features), point.label))
    metrics(predictVsRealNT2)

    println
    println("Prediction for users with 'current' loan status using DecisionTree")
    val predictionForCurrentUsers = cachedCurrentData.map(point => model2NT.predict(point.features))
    val lateLabel = loanStatus.indexOf(normalize("Late"))
    val koCount = predictionForCurrentUsers.filter(_ == lateLabel).count()
    val okCount = predictionForCurrentUsers.count() - koCount

    println
    println("Number of members that will probably repay their loans = " + okCount + " / " + predictionForCurrentUsers.count())
    println("Number of members that will probably not repay their loans with respected delay = " + koCount + " / " + predictionForCurrentUsers.count())

    println

    println(" ------------------------  begin cleaning : only 4 features (dti, loan_amount, funded_amount, annual_income) ----------------------------------- ----------------------")
    val cachedData4C = normalizeData_4C(data)
    println("end cleaning\ndata count : " + cachedData4C.count())

    println("")
    val splitedData4C = cachedData4C.randomSplit(Array(0.70, 0.30), seed = 25896)
    val trainingData4C = splitedData4C(0)
    val testData4C = splitedData4C(1)
    println()
    // LogisticRegressionWithLBFGS
    println("Start LogisticRegressionWithLBFGS ...")
    val model14C = new LogisticRegressionWithLBFGS().setNumClasses(loanStatus.length).run(trainingData4C)
    val predictVsReal4C = testData4C.map(point => (model14C.predict(point.features), point.label))
    metrics(predictVsReal4C)

    println()
    categoricalFeaturesInfo = Map[Int, Int]()

    // DecisionTreeModel
    println("Start DecisionTreeModel ...")
    val model24C = DecisionTree.trainClassifier(trainingData4C, numClasses = loanStatus.length, categoricalFeaturesInfo,
      impurity = "gini", maxDepth = 5, maxBins = 32)

    val predictVsReal4C1 = testData4C.map(point => (model24C.predict(point.features), point.label))
    metrics(predictVsReal4C1)
    println

    // RandomForest
    println("Start RandomForest ...")
    val model34C = RandomForest.trainClassifier(trainingData4C, numClasses = loanStatus.length, categoricalFeaturesInfo,
      numTrees = 5, featureSubsetStrategy = "auto", impurity = "gini", maxDepth = 5, maxBins = 32)
    val predictVsReal4C2 = testData4C.map(point => (model34C.predict(point.features), point.label))
    metrics(predictVsReal4C2)
  }

  def metrics(predictVsReal: RDD[(Double, Double)]) = {
    println("Start BinaryClassificationMetrics ...")
    val metrics = new BinaryClassificationMetrics(predictVsReal)

    println("Recall")
    val recall = metrics.recallByThreshold().collect
    recall.foreach { case (label, value) => println(s"Threshold :$label, Recall : $value") }

    println("Precision")
    val precision = metrics.precisionByThreshold().collect()
    precision.foreach { case (label, value) => println(s"Threshold :$label, Precision : $value") }

    println("fMeasure")
    val fMeasure = metrics.fMeasureByThreshold().collect()
    fMeasure.foreach { case (label, value) => println(s"Threshold :$label, fMeasure : $value") }

    val r2 = recall.map { case (label: Double, recallValue: Double) => {
      val precisionValue = precision.filter { case (label1, _) => label1 == label }.map { case (_, value1) => value1 }.toList(0)
      (label, (recallValue * precisionValue) / ((recallValue + precisionValue)))
    }
    }
    println("R2 = (recall * precision) / (recall + precision)")
    r2.foreach { case (label, value) => println(s"Threshold :$label, R2 : $value") }

  }


}
