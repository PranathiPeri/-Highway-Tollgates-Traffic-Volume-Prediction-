# Highway-Tollgates-Traffic-Volume-Prediction

Introduction

Tollgates form the major bottlenecks in traffic during rush hours. Long queues at tollgates during rush hour can overwhelm traffic management authorities.  This can be avoided by following countermeasures like expediting the toll collection by opening more lanes during rush hours or streamlining the future traffic by adaptively tweaking traffic signals at upstream intersections. The prediction will allow the traffic management authorities to capitalize on big data & algorithms for fewer congestions at tollgates. These countermeasures can be deployed only when there is a reliable source of future rush hour prediction. For example, if heavy traffic is predicted in the next, then traffic regulators could open new lanes and/or divert traffic to other intersections. 
As with most prediction problems dealing with critical traffic data, we needed to explore models that give hard figures as opposed to general categories of simply ‘high’ or ‘low’ volumes. We found neural network to give the best results, as it can model most accurately the heavily nonlinear data produced by the traffic and give actual volume figures as desired. While regression could achieve good results, neural net was noticeably better. This can be attributed to its strengths in dealing with nonlinear separation boundaries, strong correlation between features, ability to implicitly detect complex nonlinear relationships between dependent and independent variables, and its ability to detect all possible interactions between predictor variables.

Problem Statement

To predict average tollgate traffic volume at each tollgate.
For every 20-minute time window, we predict the entry and exit traffic volumes at tollgates a target area with tollgate numbers 1, 2 and 3 .We predict the traffic volume of entries and exits to and from a tollgate separately. 

Data Description

The dataset we are using for traffic volume prediction is taken from KDD cup  which can be found  at  KDD dataset Link .We are using two data sets “Weather data” and “Traffic Volume through the Tollgates”. Traffic flow patterns vary due to different stochastic factors, such as weather conditions, time of the day, etc.  

Tools and Languages Used

To construct the model, we used Spark engine with python. We used Python to code the neural net. We also used Hadoop distributed file system to store the data for use. We used:
•	pyspark.sql module for creating DataFrame, register DataFrame as tables, execute SQL over tables. 
•	matplotlib.pyplot to visualize data.


Best Results:
We obtained the best results using the following parameters:
•	tanh as the sigmoid function
•	16 hidden layers with 3 neurons in each
We first tested by doing a 80/20 split of training data which gave us an MAPE error of 0.013405. We then tried our model on the derived test data which gave an MAPE error of 0.041021.


