
Script documentation for file: examine_groundtruth_comparisons, Updated on:2020-09-03 16:55:38.406363
=====================================================================================================

# Summary


This script assumes that one has already generated comparisons between raw auto-detected pursuits and groundtruth labeled by Zahra using the `data.calculate_statistics()` method. It then takes the generated comparisons, and calculates the percentage of groundtruth pursuits that are correctly captured by raw auto-detection and the percentage of auto-detections that do not correspond to a groundtruth labeled pursuit events.  These represent the true detection rate and false detection rate, respectively. It is important to note that as a hyperparameter we include the a 'buffer': when evaluating if pursuit events from one set of criteria (auto-detection or groundtruth) are found by another, we allow for a frame buffer to capture slight offsets. The default value of this buffer is 5 frames on each side. 
# Data Overview


Here we present the raw/groundtruth comparison data, divided into experiments, rois, and 40 minute segments.  

### TempTrial2

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.0|1.0|0.8666666666666667|0.6|1.0|0.6842105263157895|
|1|None|None|0.6666666666666666|0.5121951219512195|1.0|0.5|
|2|1.0|0.875|0.9090909090909091|0.1891891891891892|0.875|0.725|
|3|None|None|1.0|0.23076923076923078|1.0|0.875|
|4|1.0|0.5|1.0|0.4666666666666667|None|None|
|5|None|None|None|None|0.8571428571428571|0.625|
|6|1.0|0.875|None|None|1.0|0.8888888888888888|
|7|None|None|None|None|None|None|
|8|None|None|1.0|0.0|None|None|
  

### TempTrial7

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.72|0.581081081081081|0.6153846153846154|0.7|0.7777777777777778|0.782608695652174|
|1|0.6666666666666666|0.5925925925925926|0.25|0.9333333333333333|1.0|0.9230769230769231|
|2|0.6666666666666666|0.75|None|None|None|None|
|3|0.75|0.7727272727272727|None|None|1.0|0.4|
|4|None|None|None|None|0.0|1.0|
|5|0.0|1.0|0.5|0.8|None|None|
|6|None|None|1.0|0.5|None|None|
|7|None|None|None|None|None|None|
|8|None|None|None|None|None|None|
  

### TempTrial8

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.8571428571428571|0.7142857142857143|1.0|0.7948717948717948|1.0|0.8947368421052632|
|1|1.0|0.6486486486486487|0.6666666666666666|0.8333333333333334|1.0|0.5|
|2|1.0|0.46153846153846156|1.0|0.5714285714285714|None|None|
|3|1.0|0.8235294117647058|1.0|0.6666666666666666|None|None|
|4|None|None|1.0|0.8181818181818182|1.0|0.8|
|5|None|None|1.0|0.42105263157894735|None|None|
|6|None|None|None|None|None|None|
|7|None|None|None|None|None|None|
|8|None|None|None|None|None|None|
  

### TempTrial9

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.875|0.7884615384615384|0.6666666666666666|0.8148148148148148|1.0|0.9411764705882353|
|1|1.0|0.88|None|None|1.0|0.4444444444444444|
|2|None|None|None|None|None|None|
|3|None|None|None|None|None|None|
|4|None|None|None|None|0.9090909090909091|0.25|
|5|None|None|None|None|None|None|
|6|None|None|None|None|None|None|
|7|None|None|None|None|None|None|
|8|None|None|None|None|None|None|
  

### TempTrial10

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|None|None|0.6956521739130435|0.2727272727272727|0.8125|0.72|
|1|None|None|1.0|0.2413793103448276|0.625|0.9322033898305084|
|2|None|None|1.0|0.7777777777777778|0.8|0.9518072289156626|
|3|None|None|None|None|0.5|0.9722222222222222|
|4|None|None|1.0|0.0|None|None|
|5|None|None|None|None|None|None|
|6|None|None|None|None|None|None|
|7|None|None|1.0|0.5|0.0|1.0|
|8|None|None|None|None|None|None|
  
