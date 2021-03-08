# Testing RPCA based foreground segmentation on Froemke/Carcea Data. 

I ran a preliminary test of RPCA using the implementation of Principal Component Pursuit provided by the onlienRPCA package. As an initial run, I just provided four training frames (frames 1-4) of the training data, each of dimensionality 410x300x3. We created a data matrix M of shape (410x300x3,4), and applied RPCA (scripts/rpca_experiment.py). The results can be seen in the image below:  
<img src="./images/RPCA_reconstruction.png" />

On the left, we see that the recovered sparse components are able to accurately segment out the locations of the mice in each training frame, if not their appearance (after all, we are considering this as an additive model). We see that the learned low dimensional components actually include all possible locations of the mice, and prefer to learn the sparse components as subtractions. This is likely because the mice themselves are so close to black that the optimization works better this way. We might want to consider using this with a 0-1 flipped image instead if this is going to be annoying. It would be interesting to see if the solution switches to considering the true static background when we have enough frames.   

Interesting future directions: 
- [ ] How do we make this faster? Look into the online implementations, svd solver (https://github.com/wxiao0421/onlineRPCA). 
    - [ ] The code for the offline implementations is done entirely with numpy. This makes it available for just in time compilation, and potentially gpu acceleration via libraries like JAX and numba. This appears to be pretty simple, and GPU accelerated SVD in particular should be available through JAX.   
- What if we only use one channel? Could this further speed up analysis without hurting performance?  
- How does this compare to neural network approaches? (speed, performance)
- Is there room for improvement in what people have considered?  

Update 3/8, 12:34
It turns out that accelerating this computation by using GPUs is not necessarily a good option. It looks like using GPUs we run into OOM issues very quickly, while using the 8 cpus on the p3.2 machine we actually do get a substantial increase in performance. We can revise our future directions as follows: 
- multi core parallelization through numpy seems like quite a good option. We can probably extend this by using JAX just in time compulation on the update step internally.  
- How much does the multi-core benefit help us? We should time this. 



