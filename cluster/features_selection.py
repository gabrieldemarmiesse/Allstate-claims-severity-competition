# coding=utf-8

from __future__ import print_function
def print(*arg):
    mystring = ""
    for argument in arg:
        mystring += str(argument)
    f = open('log.txt', 'a')
    f.write(mystring + "\n")
    f.close()

# Initialize SparkContext
import sys
from pyspark import SparkContext
from pyspark import SparkConf
sc = SparkContext()
import os
import sys
import re
from pyspark import SparkContext
from pyspark import SparkContext
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
from pyspark.sql import types
from pyspark.sql import Row
from pyspark.sql import functions
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import numpy as np
import pyspark.sql.functions as func
import matplotlib.patches as mpatches
import time as time
from matplotlib.patches import Rectangle
import datetime
import ast
from operator import add
import math
from itertools import combinations
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml import Pipeline
from pyspark.ml.regression import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import random

print("test with 3 trees and 78 features, depth 18")

df = sqlContext.createDataFrame(sc.pickleFile("rdd1.p", 30), ["label", "cat_features","cont_features"]).cache()

to_delete = [2,3,4,5,6,7,8,96]    # cela correspond à cat2, cat3, cat4 ...
features_to_keep = list(range(116))
for idx in to_delete:
    features_to_keep.remove(idx - 1) # Car to_delete commence à 1

class customTransformer:
    
    def __init__(self, inputCol, outputCol, *others):
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.args = list(others)
        self.fitInfo = 0
        
    # Store information taken from the dataframe
    def fit(self, df, fitting):
        idx = df.columns.index(self.inputCol)
        self.fitInfo = fitting(df.rdd.map(lambda x: x[idx]))
        return self
            
    # This transforms a dataframe into another dataframe
    def transform(self, df, transforming):
        
        # We get the index of the colmumns we'll be working on
        names = df.columns
        idx = names.index(self.inputCol)
        
        # We apply the transformation
        bInfo = sc.broadcast(tuple([0,[10]]))
        new_column = df.rdd.map(lambda x: transforming(x[idx], bInfo.value[0], bInfo.value[1]), True)
        
        # We attach the results to the old rdd
        old_rdd = df.rdd.map(lambda x: list(x))
        new_rdd = old_rdd.zip(new_column).map(lambda x: x[0] + [x[1]])
        new_names = names + [self.outputCol]
        
        return sqlContext.createDataFrame(new_rdd, new_names)


def apply(df, listOfTransformers):
    df1 = df
    for transformer in listOfTransformers:
        df1 = transformer.transform(df1)
    return df1
	
def t_bucketize(element, fitInfo = 0, args = []):
    new_tup = tuple(int(scalar*args[0]) for scalar in element)
    return Vectors.dense(new_tup)
	
# Pre-process
df1 = customTransformer("cont_features", "cont_bucked_features", 7).transform(df, t_bucketize).cache()
df.unpersist()

feat_cat = [118, 79, 116, 14, 23, 61, 6, 57, 109, 59, 121, 78, 54, 99, 68, 47, 119, 117, 15, 49, 45, 124, 126, 97, 0, 127, 76, 129, 80, 81, 25, 114, 36, 92, 91, 58, 103, 104, 18, 65, 113, 128, 7, 27, 100, 101, 82, 20, 106, 53, 122, 13, 72, 105, 40, 9, 108, 115, 35]
feat_cont = [a - 116  for a in feat_cat if a>=116]
feat_cat = [a for a in feat_cat if a<116]
# We now have 10 set of features to test
sets_cat_features = [sorted(random.sample(feat_cat,40)) for _ in range(10)]
sets_cont_features = [sorted(random.sample(feat_cont,4)) for _ in range(10)]
sets_features = list(zip(sets_cat_features,sets_cont_features))

# Defining the transformations
indexer_cat = VectorIndexer(inputCol="cat_features", outputCol="cat_indexedFeatures", maxCategories=300).fit(df1)
indexer_cont = VectorIndexer(inputCol="cont_bucked_features", outputCol="cont_indexedFeatures", maxCategories=7).fit(df1)


rf = RandomForestRegressor(labelCol="label", featuresCol="features", maxBins=300,\
                            subsamplingRate=0.9, numTrees=3)

df2 = apply(df1,[indexer_cat, indexer_cont]).cache()
df1.unpersist()

for set_features in sets_features:

    slicer_cat=VectorSlicer(inputCol="cat_indexedFeatures", outputCol="sliced_cat_Features", indices=set_features[0])
    slicer_cont=VectorSlicer(inputCol="cont_indexedFeatures", outputCol="sliced_cont_Features", indices = set_features[1])
    assembler = VectorAssembler(inputCols=["sliced_cat_Features", "sliced_cont_Features"], outputCol="features")
    pipeline = Pipeline(stages=[slicer_cat, slicer_cont, assembler,rf])
    # defining the parameters to test
    paramGrid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [18]) \
        .build()

    myRegressor = RegressionEvaluator(metricName="mae")

    # defining the cross-validation
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=myRegressor,
                              numFolds=3)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(df2)
    print("features: " + str(set_features))
    print(cvModel.avgMetrics[0]/3)
    print(myRegressor.evaluate(cvModel.bestModel.transform(df2)))
    print(cvModel.bestModel.stages)
	




