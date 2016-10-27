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
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.linalg.distributed import RowMatrix

input_path = "./train.csv"
raw_data = sc.textFile(input_path , 30)
print("number of partitions ", raw_data.getNumPartitions())

# extract the header
header = raw_data.first()

#raw_data = raw_data.repartition(18)


# replace invalid data with NULL and remove header
cleaned_data = raw_data.filter(lambda row: row != header)

print("number of rows after cleaning:", raw_data.count())
print("Number of partitions: " + str(raw_data.getNumPartitions()))

sqlContext = SQLContext(sc)

names = header.split(";")
cats = names[1:117]
conts = names[117:-1]
def create_StructField(string):
    hint = string[:3]
    if hint == "cat":
        datatype = types.IntegerType()
    elif hint == "con":
        datatype = types.FloatType()
    elif hint == "id":
        datatype = types.IntegerType()
    elif hint == "los":
        datatype = types.FloatType()
    else:
        raise ValueError("Can\'t read this string:" + hint )

    return types.StructField(string, datatype, False)
	
structField_list = [create_StructField(string) for string in names]

data_schema = types.StructType(structField_list)

def tryeval(val,column_number):
    if column_number == 0:
        return int(val)
    elif 1 <= column_number <= 116:
        return val
    elif 117 <= column_number <= 131:
        return float(val)
    else:
        raise Exception("There is a big problem")

def to_tuple(string, character = ";"):
    list_of_strings = string.split(character)
    return tuple(tryeval(string, n) for n, string in enumerate(list_of_strings))

cleaned_data_splitted = cleaned_data.map(lambda x: to_tuple(x))

def to_tuples(list_):
    return tuple((string,) for string in list_)

def fusion(x, y):
    return tuple(tuple(set(xi + yi)) for xi, yi in zip(x,y))

list_of_dictionaries = []
a = cleaned_data_splitted.map(lambda x: to_tuples(x[1:117])).reduce(fusion)

sorted_tuples = tuple(tuple(sorted(tup)) for tup in a)

for tup in sorted_tuples:
    my_dict = dict()
    for idx, cat in enumerate(tup):
        my_dict[cat] = idx
    list_of_dictionaries.append(my_dict)
	
bListOfDictionaries = sc.broadcast(list_of_dictionaries)

def replace(row):
    strings = row[1:117]
    my_dicts = bListOfDictionaries.value
    tuple_of_ints = ()
    for dict_, string in zip(my_dicts, strings):
        try:
            tuple_of_ints += (dict_[string],)
        except KeyError:
            tuple_of_ints += (0,)
    return (row[0],) + tuple_of_ints + row[117:]
	
final_rdd = cleaned_data_splitted.map(replace).cache()


df = sqlContext.createDataFrame(final_rdd, schema = data_schema).coalesce(12).cache()

print(df.rdd.first())
print(final_rdd.getNumPartitions())
