# Testing RPCA based foreground segmentation on Froemke/Carcea Data. 

I ran a preliminary test of RPCA using the implementation of Principal Component Pursuit provided by the onlienRPCA package. As an initial run, I just provided four training frames (frames 1-4) of the training data, each of dimensionality 410x300x3. We created a data matrix M of shape (410x300x3,4), and applied RPCA (scripts/rpca_experiment.py). The results can be seen in the image below:  
<img src="./images/RPCA_reconstruction.png" />

On the left, we see that the recovered sparse components are able to accurately segment out the locations of the mice in each training frame, if not their appearance (after all, we are considering this as an additive model). We see that the learned low dimensional components actually include all possible locations of the mice, and prefer to learn the sparse components as subtractions. This is likely because the mice themselves are so close to black that the optimization works better this way. We might want to consider using this with a 0-1 flipped image instead if this is going to be annoying. It would be interesting to see if the solution switches to considering the true static background when we have enough frames.   

Interesting future directions: 
- How do we make this faster? Look into the online implementations, svd solver (https://github.com/wxiao0421/onlineRPCA). 
- What if we only use one channel? Could this speed up analysis without hurting performance?  
- How does this compare to neural network approaches? (speed, performance)
- Is there room for improvement in what people have considered?  


