from script_doc_utils import *
import skimage.filters as filters
import joblib
from joblib import Memory
import jax
from jax.experimental import optimizers
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from social_pursuit.labeled import LabeledData

labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"
fixture_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir")


cachedir = "/Volumes/TOSHIBA EXT STO/cache"
memory = Memory(cachedir, verbose=1)

@memory.cache
def optimize_quadratic():
    """ 
    Optimize a quadratic function. 
    """
    def loss(x):
        return x**2

    grad_func = jax.grad(loss)

    step_size = 0.01
    x_init = np.random.randn()
    x = x_init
    all_x = []
    all_val = []
    iterations = 100
    for i in range(iterations):
        print("value: {} iteration: {}".format(x,i))
        step = grad_func(x)*step_size
        x -= step
        all_x.append(x)
        all_val.append(loss(x))

    plt.plot(all_val)
    plt.title("Function value")
    fig = plt.gcf()
    return fig,all_val,all_x,iterations

def gaussian(xy,mean,sigma=10):
    """Returns a function that gives the gaussian likelihood at evaluation points. 
    :param mean: an x,y tuple containing the mean of the mean. 
    :param sigma: a scalar giving the variance of the gaussian filter (1 by default)

    """
    nll = jnp.linalg.norm(mean-xy)**2/(2*sigma) 
    return jnp.exp(-nll)

def place_gaussian(coordinates,dims):
    """Places a gaussian convolution on a 2d array with shape given by dims. 

    """
    assert len(dims) == 2
    assert len(coordinates) == 2
    mapped_gaussian = jax.vmap(lambda xy: gaussian(xy,mean = jnp.array(coordinates)))
    x1,x2 = jnp.arange(dims[0]),np.arange(dims[1])
    X1,X2 = jnp.meshgrid(x1,x2,indexing = 'ij')
    coords = jnp.stack([X1.flatten(),X2.flatten()],axis = 1)
    output = mapped_gaussian(coords,)
    return output.reshape(dims)

place_gaussians = jax.vmap(place_gaussian,(0,None),0)

@memory.cache
def optimize_image_intensity():
    """Find the location of maximum image intensity on the image. 

    """

    def image_loss(coordinates,image):
        """Given a set of coordinates, calculates the loss against the image as the negative dot product of the gaussian convolution with the image.  

        """
        dims = image.shape
        conv_im = place_gaussian(coordinates,dims) 
        total = jnp.sum(image*conv_im)
        return -total

    image_gradfunc = jax.grad(image_loss,argnums =0)

    step_size =5 
    x_init = np.random.randn()+np.array([40,72])
    x = x_init
    all_x = []
    all_val = []
    iterations = 200
    for i in range(iterations):
        print("value: {} iteration: {}".format(x,i))
        sobel = filters.sobel(image_channel0)
        step = image_gradfunc(x,sobel)*step_size
        x -= step
        all_x.append(x)
        all_val.append(image_loss(x,sobel))

    all_x = jnp.array(all_x)

    fig,ax = plt.subplots(2,1)
    ax[0].imshow(filters.sobel(image_channel0))
    ax[0].plot(all_x[:,1],all_x[:,0],"r")
    ax[0].plot(all_x[-1][1],all_x[-1][0],"ro")
    ax[1].imshow(place_gaussian(all_x[-1],image_channel0.shape))
    ax[0].set_title("Optimization trajectory on image")
    ax[1].set_title("Convolved search object")
    return plt.gcf()

def weight_to_fft(weightvec,pcaobject):
    """transforms a set of elliptic pca weights into a contour. 

    """
    vecs = pcaobject.components_
    reconstruct = jnp.matmul(vecs.T,weightvec)+pcaobject.mean_
    return reconstruct 

def fft_to_contour(fft):
    """transforms a set of elliptic pca weights into a contour. 

    """
    reshaped = fft.reshape(2,-1)
    featurelength = int(reshaped.shape[-1]/2)
    reshaped_complex = reshaped[:,:featurelength]+1j*reshaped[:,featurelength:]
    xy = jnp.fft.irfft(reshaped_complex)
    return xy

def weight_to_contour(weightvec,pcaobject):
    """transforms a set of elliptic pca weights into a contour. 

    """
    fft = weight_to_fft(weightvec,pcaobject)
    xy = fft_to_contour(fft)
    return xy

@memory.cache(ignore = ["md","weightvec","trained"])
def optimize_contour_to_template(md,weightvec,trained,iterations = 3000):

    xy = weight_to_contour(weightvec,trained)
    fft = weight_to_fft(weightvec,trained)
    angle = 0.2
    rotmat = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    template = weight_to_contour(weights[10,:],trained)
    template  = np.matmul(rotmat,xy)
    plt.plot(*xy)
    plt.plot(*template)
    plt.axis("equal") 
    fig = plt.gcf()
    save_and_insert_image(md,fig,"../docs/script_docs/images/manually_generated_contour.png")

    md.new_paragraph("We are able to phrase the transformation in terms of matrix multiplication and jax operations. We now want to come up with a loss to manipulate this image into a shifted virsion of it. One brute force thing to try would just be to try and optimize the dot product of this vector with a shifted version of it. This will be a good measure of how much the fft and pca transforms make this challenging.")

    def templateloss(fft,template):
        contour = fft_to_contour(fft)
        return jnp.linalg.norm(contour.flatten()-template.flatten())

    lossgrad = jax.grad(templateloss,argnums = 0)

    fft_len = 1/np.linspace(1,len(fft)/2,len(fft)/4)
    fft_len = 1/(np.fft.rfftfreq(len(fft)//2-1))
    fft_len[0] = 1028 
    fft_len = fft_len/fft_len[1]
    #fft_len[30:] = 0
    scaling = np.concatenate([fft_len,fft_len,fft_len,fft_len])
    x_init = jnp.array(list(fft)) 

    step_size = 10#*scaling
    #opt_init,opt_update,get_params = optimizers.adam(step_size)
    #opt_state = opt_init(x_init)

    x = x_init
    all_x = []
    all_val = []
    for i in range(iterations):
        try:
            step = lossgrad(x,template)*step_size
            #opt_state = opt_update(0,step,opt_state)
            #x = get_params(opt_state)
            x -= step
            all_x.append(x)
            all_val.append(templateloss(x,template))
            print(all_val[-1])
        except KeyboardInterrupt:
            break

    all_x = jnp.array(all_x)

    contour = fft_to_contour(x)
    orig_contour = fft_to_contour(x_init)
    plt.plot(*contour,label = "trained")
    plt.plot(*template,label = "template")
    plt.plot(*orig_contour,label = "orig")
    plt.legend()
    plt.axis("equal")
    fig = plt.gcf()
    return fig

if __name__ == "__main__":
    data = LabeledData(labeled_data,additionalpath)
    md = initialize_doc({"parent":"from_contours_to_shape_model"})
    md.new_header(level = 1,title = "Experimenting with JAX for automatic differentiation.")

    md.new_paragraph("We'll work out three examples here: the first will simple minimum finding of a quadratic function, the second will be looking for intensity peaks in an image, and the third will be transforming one of our contours into a circle. These will be training/plausibility tests for an active shape model in pca fourier space.")

    md.new_header(level = 2,title = "Quadratic Function")
    md.new_paragraph("First let's try finding the minimum of a quadratic. This is the function we will optimize over:")
    quadratic = jnp.linspace(-1,1,100)**2
    plt.plot(quadratic)
    fig = plt.gcf()
    save_and_insert_image(md,fig,"../docs/script_docs/images/quadratic_function.png")

    fig,all_val,all_x,iterations = optimize_quadratic()

    save_and_insert_image(md,fig,"../docs/script_docs/images/example_evaluation_trajectory.png")


    md.new_paragraph("We see that the function reached a value of {}, with x = {} after {} iterations. This optimization was very easy to write, and involved no new api to learn.".format(all_val[-1],all_x[-1],iterations))

    md.new_header(level = 2,title = "Find intensity peaks in an image.")
    md.new_paragraph("We will next take an example image, and look at the edge intensity. While this is not a very useful example, we will traverse the image in search of intensity peaks. In order to have some more flexibility with this, we will convolve the search point with a gaussian, and array the resulting intensities in 2D. We then take the dot product of this 2D search element with the original image and feed this to the loss function.")

    image = data.get_images([10])[0]
    image_channel0 = image[:,:,0]
    plt.imshow(image_channel0)
    plt.title("Image we will be traversing")
    fig = plt.gcf()
    save_and_insert_image(md,fig,"../docs/script_docs/images/traverseimage.png")

    fig = optimize_image_intensity()
    save_and_insert_image(md,fig,"../docs/script_docs/images/optimized_image_search_element.png")

    md.new_paragraph("Here we use a gaussian convolved image of sigma = 20, and successfully localize a maximum of the sobel edge image.")

    md.new_header(level = 3,title = "Find make a contour more circular")
    md.new_paragraph("Our final check will be to take a contour object that has been parametrized as the fourier transform of pca weights, and train through the pca transformation. We can take gradients directly through the irfft transformation with JAX, but we will have to formulate the PCA transformation explicitly. ")

    fouriers = joblib.load(os.path.join(fixture_dir,"fdsampleset"))

    out = data.organize_fourierdicts(fouriers)
    nb_components = 10
    pcadict = data.get_pca(out,nb_components)

    animal = "dam"
    animaldict = {"dam":1}

    animalpca = pcadict[animal]

    weights = animalpca["weights"]
    trained = animalpca["pca"]
    components = trained.components_

    weightvec = weights[0,:]
    print(weightvec.shape)
    
    auto = trained.inverse_transform(weightvec)



    manual = np.matmul(components.T,weightvec)+trained.mean_
    assert np.all(np.abs(auto-manual)<1e-10)
    fig = optimize_contour_to_template(md,weightvec,trained)
    save_and_insert_image(md,fig,"../docs/script_docs/images/grad_descent_image.png")
    md.new_paragraph("I am learning that it's actually quite difficult to do gradient descent through a fourier transform. Initially, I tried to do gradient descent through the pca weights, optimizing the contour against a template contour that was a shifted and rotated version of it. This proved to be quite difficult. My hypothesis was that this was because having the tip of the nose off center was not a reachable position from the pca space. To address this, I tried to perform template matching to another real contour instead. This also proved too difficult. My next thought was that PCA was making certain configurations difficult to reach, so I tried scaling the step size by the eigenvalues, which did not work either. Finally I abandoned the pca and tried to take gradients through the fft. This seems to work better, but it still appears to be very sensitive. In particular, when I tried normalizing the step size by the harmonic series, this seemed to work a lot better, until I remembered that the whole thing is a concatenation of real and complex values. I then tried instead to concatenate the harmonic series scaling in a symmetric way to respect the structure of the harmonic series, but once again this looked to be flawed- because this is wrong: it should NOT be reversed. What is shown here is the version with scaled step sizes in fourier space. Although the idea of scaling the fourier components by their frequency makes sense in terms of fuzzy intuition, it's very weird, and we might want to consider alternative parametrizations in the future (consider the explicit ellipsoid parametrization by staib and duncan, for example). What is promising is that fitting to edges is actually a far more lenient task than fitting to a vectorized template, as was done here. Even if we don't take gradient steps in pc space, we can still regularize our cost function with our gaussian model likelihood to maintail plausibility." )
    md.new_paragraph("After reviewing the Staib and Duncan paper, they show that it is sufficient to do gradient descent on the same representation that we currently are doing. They mention that taking gradients through their relative ellipse based parametrization was difficult due to the complexities of the gradient derivation. This could be interesting for us.")
    md.new_paragraph("I think I finally got the step size to something that is internally consistent, by setting it equal to 1/the frequencies distributed over both x and y. This just seems like a hard optimization problem in general. I've also seen that it seems like it could be a good idea to restrict the gradients to the first few frequencies. These are all options we should keep in mind going forwards, but let's see if we can make our problem more specific. Let's try an alternative, where we convert the rotated template to a grayscale image and try to find edges there with a gaussian blurred contour as we did before. ")
    md.new_paragraph("UPDATE 10/30: it looks like this optimization works fine, even if it looks like the cost is blowing up. We don't need to scale the frame rate by training, or anything. Working hypothesis is that we started getting into weird scaling stuff too early, we should have stuck to the fft. This is sufficient evidence for me to believe that if we train the fourier descriptors by gradient descent, we should be able to get somewhere.")
    md.new_paragraph("Now we will develop these experiments into source code, and use them in the parent file (from_contours_to_shape_model.py)")


    md.create_md_file()
