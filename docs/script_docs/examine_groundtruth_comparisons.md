
Script documentation for file: examine_groundtruth_comparisons, Updated on:2020-09-18 18:58:28.231659
=====================================================================================================

# Summary


This script assumes that one has already generated comparisons between raw auto-detected pursuits and groundtruth labeled by Zahra using the `data.calculate_statistics()` method. It then takes the generated comparisons, and calculates the percentage of groundtruth pursuits that are correctly captured by raw auto-detection and the percentage of auto-detections that do not correspond to groundtruth labeled pursuit events, when measured by event, by duration, and taking directionality into account.  It is important to note that as a hyperparameter we include a 'buffer': when evaluating if pursuit events from one set of criteria (auto-detection or groundtruth) are found by another, we allow for a frame buffer to capture slight offsets. The default value of this buffer is 5 frames on each side. 
# Data Overview (Eventwise)


Here we present the raw/groundtruth comparison data, divided into experiments, rois, and 40 minute segments. We present an eventwise comparison: what are the proportion of groudntruth sheperding events that are correctly detected at any frame by automatic tracking (true detection), and what proportion of these automatically tracked events have no overlap with manual labels? (false detection)  

### TempTrial2 (Eventwise)

|Part|ROI0_true detection proportion|ROI0_false detection proportion|ROI1_true detection proportion|ROI1_false detection proportion|ROI2_true detection proportion|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.0|1.0|0.86|0.6|1.0|0.68|
|1|None|None|0.66|0.51|1.0|0.5|
|2|1.0|0.87|0.90|0.18|0.87|0.72|
|3|None|None|1.0|0.23|1.0|0.87|
|4|1.0|0.5|1.0|0.46|None|None|
|5|None|None|None|None|0.85|0.62|
|6|1.0|0.87|None|None|1.0|0.88|
|None|None|None|None|None|None|None|
|8|None|None|1.0|0.0|None|None|
|total|0.83 (Total:12)|0.88 (Total:63)|0.88 (Total:176)|0.35 (Total:213)|0.92 (Total:54)|0.70 (Total:157)|
  

### TempTrial7 (Eventwise)

|Part|ROI0_true detection proportion|ROI0_false detection proportion|ROI1_true detection proportion|ROI1_false detection proportion|ROI2_true detection proportion|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.72|0.58|0.61|0.7|0.77|0.78|
|1|0.66|0.59|0.25|0.93|1.0|0.92|
|2|0.66|0.75|None|None|None|None|
|3|0.75|0.77|None|None|1.0|0.4|
|4|None|None|None|None|0.0|1.0|
|5|0.0|1.0|0.5|0.8|None|None|
|6|None|None|1.0|0.5|None|None|
|None|None|None|None|None|None|None|
|None|None|None|None|None|None|None|
|total|0.67 (Total:92)|0.63 (Total:141)|0.54 (Total:44)|0.75 (Total:82)|0.75 (Total:24)|0.79 (Total:67)|
  

### TempTrial8 (Eventwise)

|Part|ROI0_true detection proportion|ROI0_false detection proportion|ROI1_true detection proportion|ROI1_false detection proportion|ROI2_true detection proportion|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.85|0.71|1.0|0.79|1.0|0.89|
|1|1.0|0.64|0.66|0.83|1.0|0.5|
|2|1.0|0.46|1.0|0.57|None|None|
|3|1.0|0.82|1.0|0.66|None|None|
|4|None|None|1.0|0.81|1.0|0.8|
|5|None|None|1.0|0.42|None|None|
|None|None|None|None|None|None|None|
|None|None|None|None|None|None|None|
|None|None|None|None|None|None|None|
|total|0.93 (Total:66)|0.66 (Total:143)|0.95 (Total:40)|0.71 (Total:97)|1.0 (Total:12)|0.81 (Total:53)|
  

### TempTrial9 (Eventwise)

|Part|ROI0_true detection proportion|ROI0_false detection proportion|ROI1_true detection proportion|ROI1_false detection proportion|ROI2_true detection proportion|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|0.87|0.78|0.66|0.81|1.0|0.94|
|1|1.0|0.88|None|None|1.0|0.44|
|None|None|None|None|None|None|None|
|None|None|None|None|None|None|None|
|4|None|None|None|None|0.90|0.25|
|None|None|None|None|None|None|None|
|None|None|None|None|None|None|None|
|None|None|None|None|None|None|None|
|None|None|None|None|None|None|None|
|total|0.90 (Total:22)|0.81 (Total:77)|0.66 (Total:12)|0.81 (Total:27)|0.93 (Total:30)|0.52 (Total:50)|
  

### TempTrial10 (Eventwise)

|Part|ROI0_true detection proportion|ROI0_false detection proportion|ROI1_true detection proportion|ROI1_false detection proportion|ROI2_true detection proportion|ROI2_false detection proportion|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0|None|None|0.69|0.27|0.81|0.72|
|1|None|None|1.0|0.24|0.62|0.93|
|2|None|None|1.0|0.77|0.8|0.95|
|3|None|None|None|None|0.5|0.97|
|4|None|None|1.0|0.0|None|None|
|None|None|None|None|None|None|None|
|None|None|None|None|None|None|None|
|7|None|None|1.0|0.5|0.0|1.0|
|None|None|None|None|None|None|None|
|total|nan (Total:0)|nan (Total:0)|0.79 (Total:134)|0.30 (Total:130)|0.75 (Total:96)|0.85 (Total:419)|

# Next Steps (Eventwise)


These tables show that we have a pretty high baseline rate of capturing pursuit events (around 70-90 percent for most cases). However at the same time we also have a pretty high false detection rate (around 50-80 percent for most cases). Are there any features of falsely detected pursuits that distinguish them from true detected pursuits? Likewise, are there any features of false negative pursuits that we should be selecting for? 