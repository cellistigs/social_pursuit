# Progress Diary

## 7/20/20
Construct github repo, determine data organizational structure. 

## 8/11/20
Downloaded traces run by Zahra to /Volumes/TOSHIBA\ EXT\ STO/RTTemp\_Traces. Should be 12 videos, 3 pairs each. 
Pilot data handling interface with this data.
- Make Data integrity check. 
Overnight, run a script to download videos to your local drive.  

## 9/10/20
Took up project again. Tried a fourier decomposition, which gives interesting results. It definitely gives a different characterization of errors than what we saw before, but no guarantee that this will be a good choice for anomaly detection (very good at detecting non-smoothness though; this could be useful to detect outliers regardless of windowlength.) 
- The immediate application would be to apply this method to cluster in the frequency space- cut out the middle, and look for characteristics in a uniform, featurized space. Is this feasible given our shortest trajectories are 10 frames long? If not, is this a justification for going longer?  
