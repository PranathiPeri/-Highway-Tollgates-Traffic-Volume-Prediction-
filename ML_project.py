import csv
import heapq
import StringIO
import sys
import struct
import plotly
import os
import math
import random
import re
import numpy as np
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lag, col
from pyspark.sql.window import Window
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import dense_rank
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, udf
from IPython.core.display import display
import plotly.plotly as py
from plotly.graph_objs import *
import requests
from pyspark.sql.functions import *
from pyspark.sql.types import DateType
from datetime import datetime,timedelta
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.rdd import RDD
from pyspark.sql.types import *
from pyspark.sql.types import StringType
from pyspark.sql import SQLContext
from py4j.compat import unicode
from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JJavaError, Py4JError
from py4j.tests.java_gateway_test import PY4J_JAVA_PATH, safe_shutdown
conf = SparkConf()
conf.setMaster('spark://nithin-HP-Pavilion-15-Notebook-PC:7077')
conf.setAppName('spark-basic')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
lines = sc.textFile("/home/nithin/Desktop/project_dataSets/scripts/scripts/train_volume.csv")
parts = lines.map(lambda l: l.split(","))
weather_lines = sc.textFile("/home/nithin/Desktop/project_dataSets/weather.csv")
weather_parts = weather_lines.map(lambda l: l.split(","))
training_data = parts.map(lambda p: Row(
	tollgate_id=p[0],
	time_window = p[1],
	direction=p[2],
	volume=p[3]
	))
weather_data = weather_parts.map(lambda p: Row(
	date=p[0],
	hour = p[1],
	pressure=p[2],
	sea_pressure=p[3],
	wind_direction=p[4],
	wind_speed=p[5],
	temparature=p[6],
	rel_humidity=p[7],
	precipitation=p[8]
	))
train = sqlContext.createDataFrame(training_data)
weather = sqlContext.createDataFrame(weather_data)
train = train.withColumn('volume',regexp_replace('volume','\"','').cast(DoubleType()))
train = train.withColumn('direction',regexp_replace('direction','\"','').cast(DoubleType()))
train = train.withColumn('tollgate_id',regexp_replace('tollgate_id','\"','').cast(DoubleType()))
weather = weather.withColumn('precipitation',regexp_replace('precipitation','\"','').cast(DoubleType()))
weather = weather.withColumn('wind_direction',regexp_replace('wind_direction','\"','').cast(DoubleType()))
weather = weather.withColumn('pressure',regexp_replace('pressure','\"','').cast(DoubleType()))
weather = weather.withColumn('rel_humidity',regexp_replace('rel_humidity','\"','').cast(DoubleType()))
weather = weather.withColumn('sea_pressure',regexp_replace('sea_pressure','\"','').cast(DoubleType()))
weather = weather.withColumn('temparature',regexp_replace('temparature','\"','').cast(DoubleType()))
weather = weather.withColumn('wind_speed',regexp_replace('wind_speed','\"','').cast(DoubleType()))
train = train.withColumn("date1",train.time_window[3:10])
train = train.withColumn("hour1",train.time_window[14:2].cast(DoubleType()))
train = train.withColumn("min1",train.time_window[17:2].cast(DoubleType()))
weather = weather.withColumn('date',regexp_replace('date','\"',''))
weather = weather.withColumn('hour',regexp_replace('hour','\"','').cast(DoubleType()))
weather_pandas=weather.toPandas()
#plt.figure()
#plt.boxplot(weather_pandas['wind_direction'])
weather_pandas['rank']=weather_pandas['date'].rank(ascending=1)
weather_pandas.plot('rank','wind_direction',kind='scatter')
plt.show()
weather_pandas.boxplot(column='wind_direction')
plt.show()
weather_pandas.boxplot(column='pressure')
plt.show()
weather_pandas.boxplot(column='rel_humidity')
plt.show()
weather_pandas.boxplot(column='sea_pressure')
plt.show()
weather_pandas.boxplot(column='temparature')
plt.show()
weather_pandas.boxplot(column='wind_speed')
plt.show()
for i in range(len(weather_pandas['wind_speed'])):
	if(weather_pandas['wind_direction'][i]==999017):
		weather_pandas['wind_direction'][i]=(weather_pandas['wind_direction'][i-1]+weather_pandas['wind_direction'][i+1])/2
		print(weather_pandas['wind_direction'][i])
#plt.boxplot(weather_pandas['wind_direction'],0,'gD')
#plt.show()
weather_pandas.boxplot(column='wind_direction')
plt.show()
weather_pandas.plot('rank','wind_direction',kind='scatter')
plt.show()
weather = sqlContext.createDataFrame(weather_pandas)
join_df=train.join(weather,(train.date1 == weather.date)&((train.hour1==weather.hour)|(train.hour1==weather.hour+1)|(train.hour1==weather.hour+2)))
join=join_df.rdd
#join.saveAsTextFile("/home/nithin/Desktop/project_dataSets/preprocess_test")

join_df.registerTempTable("join_df")
get_required=sqlContext.sql("""SELECT direction, time_window, tollgate_id, volume,date1,hour1,min1,precipitation, pressure, sea_pressure, rel_humidity, wind_direction, wind_speed, temparature FROM join_df""")
get_required_pandas=get_required.toPandas()
cols_to_norm = ['volume','precipitation','pressure','sea_pressure','rel_humidity','wind_direction','wind_speed','temparature']
get_required_pandas[cols_to_norm] = get_required_pandas[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
normalized_data=sqlContext.createDataFrame(get_required_pandas)
w = Window().partitionBy().orderBy("tollgate_id","direction","time_window")
normalized_data1=normalized_data.select("*", lag("volume").over(w).alias("new_col")).na.drop()
#normalized_data.withColumn("rank", dense_rank().over("time_window"))
#print(normalized_data1.rdd.collect())
w = Window().partitionBy("hour1","min1").orderBy("tollgate_id","direction","date1")
normalized_data2=normalized_data1.select("*", lag("volume").over(w).alias("new_col2")).na.drop()
normalized_data2.registerTempTable("normalized_data2")
final_processed_data = sqlContext.sql(""" SELECT tollgate_id, precipitation, pressure, sea_pressure, rel_humidity, wind_direction, wind_speed, temparature, new_col, new_col2, volume FROM normalized_data2""")
normalized_data2=final_processed_data.rdd
def toCSVLine(data):
  return ','.join(str(d) for d in data)

lines = normalized_data2.map(toCSVLine)

#lines.saveAsTextFile('/home/nithin/Desktop/test')
# Load training data




neurons=list()
input_file='/home/nithin/Desktop/test/part-00000'
error_tol = 0.1
train_per = 80
no_of_layers = 2
for k in range(no_of_layers):
    neurons = neurons + [3]

infile = input_file
fh = csv.reader(open(infile))
data = list()
train_data = list()
test_data = list()
eta=0.1
error=1.0
counter=0

#Sigmoid Function
def sigmoid(x):
    try:    
        sig = 1.0/(1+math.exp(-x))
        return sig
    except OverflowError:
        if(x>0):
            return 1
        else:
            return 0

for row in fh:
    if("\n" not in row):
        data.append(row)
#selecting training_set and test set
train_per = train_per/100.0
train = int(train_per*len(data)+0.5)
random.shuffle(data)
train_data = data[:train]
test_data = data[train:]

input_train=list()
target_train=list()

input_test=list()
target_test=list()

for row in train_data:
    input_train.append(row[:-1])
    target_train.append(row[len(data[0])-1])

for row in test_data:
    input_test.append(row[:-1])
    target_test.append(row[len(data[0])-1])


weights=list()
input=len(input_train[0])+1


#generating random weights for hidden layer
for i in range(no_of_layers):
    output=neurons[i];
    temp=list()
    for j in range(input*output):
        temp.append(random.uniform(0,1))
    weights.append(temp)
    input=output+1
temp=list()
output=1
#generating random weights for output layer
for j in range(input*output):
    temp.append(random.uniform(0,1))
weights.append(temp)

#Back propagation Algorithm for training_set
while((error>error_tol)&(counter<2000)):
    
    counter+=1
    error=0.0
    #Forward propagation
    for z in range(len(input_train)):
        temp_weights=list()
        temp_train = list()
        output_result=list()
        delta = list()
        a=1
        temp_train.append([a]+input_train[z])
        output_result.append(temp_train)
        temp_train=np.array(temp_train)
        for k in range(no_of_layers):        
            for i in range(neurons[k]):
                temp_weights.append(weights[k][(i*(len(weights[k])/neurons[k])):(i+1)*(len(weights[k])/neurons[k])])
            temp_weights=np.array(temp_weights)
            temp_train=temp_train.astype(np.float)
            temp_train=np.transpose(temp_train)
            temp_result=np.matmul(temp_weights,temp_train)
            temp_result=np.transpose(temp_result)
            temp_result=temp_result.tolist()
            temp_result=temp_result[0]
            net_result=list()
            for i in range(len(temp_result)):
                b=(sigmoid(temp_result[i]))
                net_result=net_result+[b]
            temp_weights=list()
            temp_train = list()
            temp_train.append([a]+net_result)
            output_result.append(temp_train)
            temp_train=np.array(temp_train)
        temp_weights.append(weights[k+1])    
        temp_weights=np.array(temp_weights)
        temp_train=temp_train.astype(np.float)
        temp_train=np.transpose(temp_train)
        temp_result=np.matmul(temp_weights,temp_train)
        temp_result=np.transpose(temp_result)
        temp_result=temp_result.tolist()
        temp_result=temp_result[0]
        net_result=list()
        for i in range(len(temp_result)):
           b=(sigmoid(temp_result[i]))
           net_result=net_result+[b]
        temp_delta=(net_result[0])*(1-net_result[0])*(float(target_train[z])-net_result[0])
        #training set error calculation
        error+=((float(target_train[z])-net_result[0])*(float(target_train[z])-net_result[0])*0.5/len(input_train))
        delta.append(temp_delta)
        #backward propagation
        for j in range(no_of_layers):
            temp1_delta=list()
            new_weights=list()
            for d in range(neurons[len(neurons)-j-1]+1):
                h=weights[len(weights)-j-1]
                temp_h = list()
                temp_delta=list()
                
                for i in range(len(h)/len(delta)):
                    temp_h.append(h[(i*(len(delta))):(i+1)*(len(delta))])
                temp_h=np.array(temp_h)
                temp_h=np.transpose(temp_h)
                temp_delta.append(delta)
                temp_delta=np.array(temp_delta) 
                h1=np.matmul(temp_delta,temp_h)
                h1=h1.tolist()
                h1=h1[0][d]
                temp_delta1=(output_result[len(output_result)-j-1][0][d])*(1-(output_result[len(output_result)-j-1][0][d]))*(float(h1))
                temp1_delta.append(temp_delta1)
            for d2 in range(len(delta)):
                for d1 in range(neurons[len(neurons)-j-1]+1):
                    temp_new_weight=eta*delta[d2]*(output_result[len(output_result)-j-1][0][d1])
                    new_weights.append(temp_new_weight)
            for i in range(len(weights[len(weights)-j-1])):
                weights[len(weights)-j-1][i]=weights[len(weights)-j-1][i]+new_weights[i]
            delta=temp1_delta[1:]
        for d2 in range(len(delta)):
            for d1 in range(len(input_train[0])):
                temp_new_weight=eta*delta[d2]*(float(input_train[z][d1]))
                new_weights.append(temp_new_weight)
        for i in range(len(weights[0])):
                weights[0][i]=weights[0][i]+new_weights[i]
        
error_test=0.0 
#back propagation algorithm for test_set   
for z in range(len(input_test)):
    temp_weights=list()
    temp_train = list()
    output_result=list()
    delta = list()
    a=1
    temp_train.append([a]+input_test[z])
    output_result.append(temp_train)
    temp_train=np.array(temp_train)
    for k in range(no_of_layers):        
        for i in range(neurons[k]):
            temp_weights.append(weights[k][(i*(len(weights[k])/neurons[k])):(i+1)*(len(weights[k])/neurons[k])])
        temp_weights=np.array(temp_weights)
        temp_train=temp_train.astype(np.float)
        temp_train=np.transpose(temp_train)
        temp_result=np.matmul(temp_weights,temp_train)
        temp_result=np.transpose(temp_result)
        temp_result=temp_result.tolist()
        temp_result=temp_result[0]
        net_result=list()
        for i in range(len(temp_result)):
            b=(sigmoid(temp_result[i]))
            net_result=net_result+[b]
        temp_weights=list()
        temp_train = list()
        temp_train.append([a]+net_result)
        output_result.append(temp_train)
        temp_train=np.array(temp_train)
    temp_weights.append(weights[k+1])    
    temp_weights=np.array(temp_weights)
    temp_train=temp_train.astype(np.float)
    temp_train=np.transpose(temp_train)
    temp_result=np.matmul(temp_weights,temp_train)
    temp_result=np.transpose(temp_result)
    temp_result=temp_result.tolist()
    temp_result=temp_result[0]
    net_result=list()
    for i in range(len(temp_result)):
        b=(sigmoid(temp_result[i]))
        net_result=net_result+[b]
    temp_delta=(net_result[0])*(1-net_result[0])*(float(target_train[z])-net_result[0])
    #test set error calculation
    error_test+=((float(target_test[z])-net_result[0])*(float(target_test[z])-net_result[0])*0.5/len(input_test))
    delta.append(temp_delta)

for i in range(no_of_layers):
    print("")
    print("Hidden Layer"+str(i+1)+":")
    print('\t'),
    for j in range(neurons[i]):
        print ("Neuron"+str(j+1) + ":"),
        print weights[i][j*len(weights[i])/neurons[i]:(j+1)*len(weights[i])/neurons[i]]
        print("\t"),
i+=1
print("")
print("Output Layer"+":")
print('\t'),

print ("Neuron1"+ ":"),
print weights[i]


print ("train error:"),
print (error)

print ("cross validation error:"),
print (error_test)


















#############Testing ############################
lines = sc.textFile("/home/nithin/Desktop/project_dataSets/scripts/scripts/test_volume.csv")
parts = lines.map(lambda l: l.split(","))
weather_lines = sc.textFile("/home/nithin/Desktop/project_dataSets/test_weather.csv")
weather_parts = weather_lines.map(lambda l: l.split(","))
training_data = parts.map(lambda p: Row(
	tollgate_id=p[0],
	time_window = p[1],
	direction=p[2],
	volume=p[3]
	))
weather_data = weather_parts.map(lambda p: Row(
	date=p[0],
	hour = p[1],
	pressure=p[2],
	sea_pressure=p[3],
	wind_direction=p[4],
	wind_speed=p[5],
	temparature=p[6],
	rel_humidity=p[7],
	precipitation=p[8]
	))
train = sqlContext.createDataFrame(training_data)
weather = sqlContext.createDataFrame(weather_data)
train = train.withColumn('volume',regexp_replace('volume','\"','').cast(DoubleType()))
train = train.withColumn('direction',regexp_replace('direction','\"','').cast(DoubleType()))
train = train.withColumn('tollgate_id',regexp_replace('tollgate_id','\"','').cast(DoubleType()))
weather = weather.withColumn('precipitation',regexp_replace('precipitation','\"','').cast(DoubleType()))
weather = weather.withColumn('wind_direction',regexp_replace('wind_direction','\"','').cast(DoubleType()))
weather = weather.withColumn('pressure',regexp_replace('pressure','\"','').cast(DoubleType()))
weather = weather.withColumn('rel_humidity',regexp_replace('rel_humidity','\"','').cast(DoubleType()))
weather = weather.withColumn('sea_pressure',regexp_replace('sea_pressure','\"','').cast(DoubleType()))
weather = weather.withColumn('temparature',regexp_replace('temparature','\"','').cast(DoubleType()))
weather = weather.withColumn('wind_speed',regexp_replace('wind_speed','\"','').cast(DoubleType()))
train = train.withColumn("date1",train.time_window[3:10])
train = train.withColumn("hour1",train.time_window[14:2].cast(DoubleType()))
train = train.withColumn("min1",train.time_window[17:2].cast(DoubleType()))
weather = weather.withColumn('date',regexp_replace('date','\"',''))
weather = weather.withColumn('hour',regexp_replace('hour','\"','').cast(DoubleType()))
weather_pandas=weather.toPandas()
#plt.figure()
#plt.boxplot(weather_pandas['wind_direction'])
weather_pandas['rank']=weather_pandas['date'].rank(ascending=1)
weather_pandas.plot('rank','wind_direction',kind='scatter')
plt.show()
weather_pandas.boxplot(column='wind_direction')
plt.show()
weather_pandas.boxplot(column='pressure')
plt.show()
weather_pandas.boxplot(column='rel_humidity')
plt.show()
weather_pandas.boxplot(column='sea_pressure')
plt.show()
weather_pandas.boxplot(column='temparature')
plt.show()
weather_pandas.boxplot(column='wind_speed')
plt.show()
for i in range(len(weather_pandas['wind_speed'])):
	if(weather_pandas['wind_direction'][i]==999017):
		weather_pandas['wind_direction'][i]=(weather_pandas['wind_direction'][i-1]+weather_pandas['wind_direction'][i+1])/2
		print(weather_pandas['wind_direction'][i])
#plt.boxplot(weather_pandas['wind_direction'],0,'gD')
#plt.show()
weather_pandas.boxplot(column='wind_direction')
plt.show()
weather_pandas.plot('rank','wind_direction',kind='scatter')
plt.show()
weather = sqlContext.createDataFrame(weather_pandas)
join_df=train.join(weather,(train.date1 == weather.date)&((train.hour1==weather.hour)|(train.hour1==weather.hour+1)|(train.hour1==weather.hour+2)))
join=join_df.rdd
#join.saveAsTextFile("/home/nithin/Desktop/project_dataSets/preprocess_test")

join_df.registerTempTable("join_df")
get_required=sqlContext.sql("""SELECT direction, time_window, tollgate_id, volume,date1,hour1,min1,precipitation, pressure, sea_pressure, rel_humidity, wind_direction, wind_speed, temparature FROM join_df""")
get_required_pandas=get_required.toPandas()
cols_to_norm = ['volume','precipitation','pressure','sea_pressure','rel_humidity','wind_direction','wind_speed','temparature']
get_required_pandas[cols_to_norm] = get_required_pandas[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
normalized_data=sqlContext.createDataFrame(get_required_pandas)
w = Window().partitionBy().orderBy("tollgate_id","direction","time_window")
normalized_data1=normalized_data.select("*", lag("volume").over(w).alias("new_col")).na.drop()
#normalized_data.withColumn("rank", dense_rank().over("time_window"))
#print(normalized_data1.rdd.collect())
w = Window().partitionBy("hour1","min1").orderBy("tollgate_id","direction","date1")
normalized_data2=normalized_data1.select("*", lag("volume").over(w).alias("new_col2")).na.drop()
normalized_data2.registerTempTable("normalized_data2")
final_processed_data = sqlContext.sql(""" SELECT tollgate_id, precipitation, pressure, sea_pressure, rel_humidity, wind_direction, wind_speed, temparature, new_col, new_col2, volume FROM normalized_data2""")
normalized_data2=final_processed_data.rdd
def toCSVLine(data):
  return ','.join(str(d) for d in data)

lines = normalized_data2.map(toCSVLine)

#lines.saveAsTextFile('/home/nithin/Desktop/test2')




neurons=list()
input_file='/home/nithin/Desktop/test/part-00000'
error_tol = 0.1
train_per = 80
no_of_layers = 2
for k in range(no_of_layers):
    neurons = neurons + [3]

infile = input_file
fh = csv.reader(open(infile))
data = list()
train_data = list()
test_data = list()
eta=0.1
error=1.0
counter=0



for row in fh:
    if("\n" not in row):
        data.append(row)
#selecting training_set and test set
test_data = data

input_test=list()
target_test=list()

for row in test_data:
    input_test.append(row[:-1])
    target_test.append(row[len(data[0])-1])

error_test=0.0 
#back propagation algorithm for test_set   
for z in range(len(input_test)):
    temp_weights=list()
    temp_train = list()
    output_result=list()
    delta = list()
    a=1
    temp_train.append([a]+input_test[z])
    output_result.append(temp_train)
    temp_train=np.array(temp_train)
    for k in range(no_of_layers):        
        for i in range(neurons[k]):
            temp_weights.append(weights[k][(i*(len(weights[k])/neurons[k])):(i+1)*(len(weights[k])/neurons[k])])
        temp_weights=np.array(temp_weights)
        temp_train=temp_train.astype(np.float)
        temp_train=np.transpose(temp_train)
        temp_result=np.matmul(temp_weights,temp_train)
        temp_result=np.transpose(temp_result)
        temp_result=temp_result.tolist()
        temp_result=temp_result[0]
        net_result=list()
        for i in range(len(temp_result)):
            b=(sigmoid(temp_result[i]))
            net_result=net_result+[b]
        temp_weights=list()
        temp_train = list()
        temp_train.append([a]+net_result)
        output_result.append(temp_train)
        temp_train=np.array(temp_train)
    temp_weights.append(weights[k+1])    
    temp_weights=np.array(temp_weights)
    temp_train=temp_train.astype(np.float)
    temp_train=np.transpose(temp_train)
    temp_result=np.matmul(temp_weights,temp_train)
    temp_result=np.transpose(temp_result)
    temp_result=temp_result.tolist()
    temp_result=temp_result[0]
    net_result=list()
    for i in range(len(temp_result)):
        b=(sigmoid(temp_result[i]))
        net_result=net_result+[b]
    #temp_delta=(net_result[0])*(1-net_result[0])*(float(target_train[z])-net_result[0])
    #test set error calculation
    error_test+=((float(target_test[z])-net_result[0])*(float(target_test[z])-net_result[0])*0.5/len(input_test))
    #delta.append(temp_delta)
print ("test error:"),
print(error_test)
