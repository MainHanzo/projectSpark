In this repo, we present the pyspark script we used during the project.
Author: Yang HUA, Antoine Mathat

## Datasets:
We have tested several datasets: mushrooms, uci news aggregator, uci wine, uci adult... We selected the most representative two 
datasets in this repo. It is not intuitive to find an appropriate dataset for our project because of the limit of cluster 
configuration of google cloud free trial: if the dataset is too small, it would be hard to see changes of performance when 
scaling our cluster; if the dataset is too heavy, the worker RAM would not be enough. That's why we have selected these two 
datasets, the dataset of mushroom is relavetively small for our cluster and we have extracted one small subset of uci news 
aggreagator(25mb out of 98mb)

## Script
The corresponding script uses the pyspark api for a machine learning process on each dataset

mushroom.py : load mushroom dataset and apply a Random Forest model

news.py : load news subset aggregator and apply a Naive Bayes model

## Results
you can find all graph of our testing results in this link:
https://docs.google.com/document/d/1B0Cqt0Zu-1gWoa7ZJwUi7KplUGI7zH7rC886_bMSOgc/edit?usp=sharing


