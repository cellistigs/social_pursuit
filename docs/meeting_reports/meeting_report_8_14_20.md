# Meeting Report for August 14th, 2020 (Carcea, Froemke, and Cunningham Labs)

## Data Analyzed:
* RT\_CohousingTemp

This dataset consists of the following videos:  
* TempTrial2\_left.mpg

* TempTrial5\_middle.mpg

* TempTrial7\_right.mpg

* TempTrial8\_right.mpg

* TempTrial9\_left.mpg

* TempTrial10\_middle.mpg

* TempTrial11\_left.mpg

* TempTrial12\_right.mpg

* TempTrial14\_left\_middle.mpg

* TempTrial15\_left\_right.mpg

* TempTrial16\_middle.mpg

* TempTrial17\_right.mpg

Note: TempTrial\_14 was excluded from this analysis because I could not locate a config file for it at the moment. It will be included in future analyses. 
Total analysis time: ~6 hours per video x 11 videos * 3 mice = 198 hours of behavior data.  

## Analysis Conducted:
* Automated video preprocessing, tracking, and custom postprocessing (SocialDatasetv2 module) as packaged on the AWS AMI carcea\_stable\_02\_04\_2020 (ami-0944e125acc999543). This portion of the analysis was performed by Zahra Adahman, who uploaded the datasets with appropriate configuration files. I did not rerun video processing on these videos myself, and started with traces that already existed for the sake of efficiency. 

* Automated extraction of pursuit events. I took the output of the automated tracking pipeline, and isolated out pursuit events where one animal is chasing another. In order to perform this isolation I found all detected pursuit times, took a window around them, and saved this data independently for easy querying. To be somewhat conservative, I further filtered pursuit events (here defined simply by the velocities of both animals) by the following criteria: 1) animals must be closer than 30 pixels at some point in the pursuit event. 2) animal speeds must not exceed 20 units/frame during the pursuit event. I did not include a criterion for being outside the nest, as nests here were fairly loose in construction, and animals could be tracked within them.   

* Single Event Analysis: Plots of each individual filtered pursuit event. I generated figures that plot the start (circle), end (cross) and trajectory of individual pursuit events that also include nest position, like so:  
![example plot](images/example_trace1.png)

* Plots of spatial distribution for pursuit events for each experiment.  

* Quantification of pursuit events over time: proportion dictated by one animal or another. I aggregated the pursuit events  

