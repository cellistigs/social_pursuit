
Script documentation for file: examine_groundtruth_comparisons, Updated on:2020-09-05 13:21:08.373170
=====================================================================================================

# Summary


This script assumes that one has already generated comparisons between raw auto-detected pursuits and groundtruth labeled by Zahra using the `data.calculate_statistics()` method. It then takes the generated comparisons, and calculates the percentage of groundtruth pursuits that are correctly captured by raw auto-detection and the percentage of auto-detections that do not correspond to a groundtruth labeled pursuit events.  These represent the true detection rate and false detection rate, respectively. It is important to note that as a hyperparameter we include the a 'buffer': when evaluating if pursuit events from one set of criteria (auto-detection or groundtruth) are found by another, we allow for a frame buffer to capture slight offsets. The default value of this buffer is 5 frames on each side. 
# Data Overview


Here we present the raw/groundtruth comparison data, divided into experiments, rois, and 40 minute segments.  

### TempTrial2

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.0|1.0|0.86|0.6|1.0|0.68|
|1|None|None|0.66|0.51|1.0|0.5|
|2|1.0|0.87|0.90|0.18|0.87|0.72|
|3|None|None|1.0|0.23|1.0|0.87|
|4|1.0|0.5|1.0|0.46|None|None|
|5|None|None|None|None|0.85|0.62|
|6|1.0|0.87|None|None|1.0|0.88|
|7|None|None|None|None|None|None|
|8|None|None|1.0|0.0|None|None|
|9|0.83|0.88|0.88|0.35|0.92|0.70|
  

### TempTrial7

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.72|0.58|0.61|0.7|0.77|0.78|
|1|0.66|0.59|0.25|0.93|1.0|0.92|
|2|0.66|0.75|None|None|None|None|
|3|0.75|0.77|None|None|1.0|0.4|
|4|None|None|None|None|0.0|1.0|
|5|0.0|1.0|0.5|0.8|None|None|
|6|None|None|1.0|0.5|None|None|
|7|None|None|None|None|None|None|
|8|None|None|None|None|None|None|
|9|0.67|0.63|0.54|0.75|0.75|0.79|
  

### TempTrial8

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.85|0.71|1.0|0.79|1.0|0.89|
|1|1.0|0.64|0.66|0.83|1.0|0.5|
|2|1.0|0.46|1.0|0.57|None|None|
|3|1.0|0.82|1.0|0.66|None|None|
|4|None|None|1.0|0.81|1.0|0.8|
|5|None|None|1.0|0.42|None|None|
|6|None|None|None|None|None|None|
|7|None|None|None|None|None|None|
|8|None|None|None|None|None|None|
|9|0.93|0.66|0.95|0.71|1.0|0.81|
  

### TempTrial9

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.87|0.78|0.66|0.81|1.0|0.94|
|1|1.0|0.88|None|None|1.0|0.44|
|2|None|None|None|None|None|None|
|3|None|None|None|None|None|None|
|4|None|None|None|None|0.90|0.25|
|5|None|None|None|None|None|None|
|6|None|None|None|None|None|None|
|7|None|None|None|None|None|None|
|8|None|None|None|None|None|None|
|9|0.90|0.81|0.66|0.81|0.93|0.52|
  

### TempTrial10

|Part|ROI0_true detection proporition|ROI0_false detection proportion|ROI1_true detection proporition|ROI1_false detection proportion|ROI2_true detection proporition|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|None|None|0.69|0.27|0.81|0.72|
|1|None|None|1.0|0.24|0.62|0.93|
|2|None|None|1.0|0.77|0.8|0.95|
|3|None|None|None|None|0.5|0.97|
|4|None|None|1.0|0.0|None|None|
|5|None|None|None|None|None|None|
|6|None|None|None|None|None|None|
|7|None|None|1.0|0.5|0.0|1.0|
|8|None|None|None|None|None|None|
|9|nan|nan|0.79|0.30|0.75|0.85|
  
