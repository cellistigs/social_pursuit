
Script documentation for file: polar_classification_validation, Updated on:2020-10-12 22:51:28.307690
=====================================================================================================
 
  
**parent file: [summary_week_10_9_20](./summary_week_10_9_20.md)**  
**prev file: [polar_classification](./polar_classification.md)**
# Summary


We want to validate our ability to detect errors in animal body part positions. In order to do this, our ground truth dataset will be a set of human labeled frames. We will look at the distribution of body part distances from this ground truth dataset, and examine how well data extracted from our sample lines up with this ground truth dataset.

The algorithm is as follows:
- Identity Resolution
    - Get Mean Position
    - Cluster Body Parts Based on Mean Position
    - Detect Points where only one animal is being detected
- Part Resolution
    - Calculate per-identity centroids
    - Calculate per-identity distance matrices
    - Filter parts with distance matrix based hypothesis testing
    - Reconstruct body centroid from accepted parts


In order to validate this algorithm we have created a dataset class, `social_pursuit.labeled.LabeledData`, that can retrieve the statistics of bodies labeled in the training data, as well as sample from the training data/generate surrogate data with simulated errors.

The next step will be to implement the variance ratio criterion with 2 and 5 clusters to the spectral representations of our data. We expect that this will give us a good indicator for when we have achieved good clustering, and when we have degenerated to detection of a single animal. We should compare this to variance ration criteria on the raw angles directly, as well as on the relevant analogue measures for the actual animal coordinates, not just the angles.