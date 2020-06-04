# ActiveUnetSegmentations

For segmenting images using unet. Essentially a pixelwise classifier.

We title it active because currently we use it in conjunction with active 
contours for both creating training data and segmenting the resulting 
network predictions.

## Installing

The installation of ActiveUnetSegmentation follows a pretty standard model for 
python modules. Clone the repository, then use pip to install the program into
a virtual environment.

### Virtual Environment

Create a virtualenv to install. *If the goal is to use an NVIDIA gpu, then
check next section for installing with cuda.*
    
    python -m venv unet-env

Clone the repositiory 

    git clone https://github.com/FrancisCrickInstitute/ActiveUnetSegmentation.git
    
This will clone the repository into a folder called ActiveUnetSegmentation

Then run the associtate pip on to the src folder that contains `setup.py`.

    unet-env/bin/pip install ActiveUnetSegmentation/src
    
This will install dependencies , but it will not include native libraries.

### Installing for use with CUDA.

It is important that the cuda version matches the cuda version used with
tensorflow. 

    python -m venv unet-cuda-env
    unet-cuda-env/bin/pip install tensorflow-gpu==1.13.1 keras scikit-image numpy urwid click
    
**Tensorflow 1.13 has worked on cuda 10.0 and works with multigpu**

The environment will be prepared, and the package can be installed as before,
but the dependencies will already be met. Then the corresponding cuda libraries
will need to be setup. 

    ml CUDA/10.0.130 cuDNN/7.5.0.56-CUDA-10.0.130

This should work with a system using module load, an alternative is the 
point LD_LIBRARY_PATH at the folders with the cuda+cudnn .so files, but the 
location of these is very system and version specific.

### Using Anaconda.

First use module to load the necessary modules.

    ml Anaconda3/2019.07

Create a conda environment with the relevant packages installed.

    conda create --name unet-3d numpy scikit-image tensorflow-gpu=1.13.1 keras click    

Once that is done, notice that it has installed all of the native libs, we can
activate the environment.

    conda activate unet-3d

This might ask conda to be initialize your shell first. 
( This is part of why I don't like Anaconda, but once it is 
initialized it works ok. )

Then I create a virtual environment, as above, but using the system packages so
everything doesn't get re-installed.

    python -m venv unet-cuda-conda --system-site-packages
    ./unet-cuda-conda/bin/pip install ActiveUnetSegmentation/src
    
That should install two new packagse in the venv, the unet library and urwid.

## Data

The data requires a paired set of images. An input image and an output image.
The easiest way to get started is to create a folder where you want to work
and create the folders, 'images' and 'labels'. Then put the input images in
image and corresponding label image in the folder labels with the same name.
They don't have to have the same name.

### Input

The input image is just a standard image that you want to label, I usually make 
sure to save it as a grayscale image to simplify the channels.

For 2D segmentations I commonly use the `input_shape` parameter of.

    input_shape (1, 3, 64, 64)

This is a single channel, that processes three time points of 64x64 patches.
In practice I use images saved from fiji as 2D time series with a grayscale
lookup table. When images are loaded images with multiple time points  
but a single z slice the time and depth are transposed.

For 3D segmetations I commonly use

    input_shape (1, 64, 64, 64)
    
The images are saved as 3D time series. When they are loaded they have the
shape (time, channel, z, y, x)

Data loading occurs in [unetsl.data](src/unetsl/data.py)

### Labels

The labelled images need to be the same shape image for now with a single channel.
When specifying  a data source, there are two main label types. multiclass labels,
or linear labels.

*Multiclass labels* split the data in to channels for the output of the model. For
out work we labelled a skeleton with membrane and vertex labels so labelled 
image that we supply as a data source would be a single channel with 1's and 3's.
Then the data gets split for the output of the model, one channel per label.

    1 0 0     1 0 0   0 0 0
    0 3 0  -> 0 1 0 & 0 1 0
    0 1 0     0 1 0   0 0 0
   
The way these labels are described work well with a sigmoid acitivation because 
the labels can be overlapping.

The multiclass labeller uses powers of 2, so a 16bit pixel format can have 16
unique labels. 1, 2, 4, 8, 16, 32, 64 would each correspond to a unique label.
Combinations can be used for overlapping labels.

*Linear labels* are the next class of label, they correspond to a 'relu' activation
and a single channel of output.

These Do not change the output so they can work with float pixels or integer type pixels. We use this type of labelling
with the distance transform. Linear labl




## Running

All of the commands shown here are installed in the virtual environment, so
either it needs to be activated, or a full path needs to be used.

The program is run in three steps. First create the model, then train the model,
then use the model to make predictions.

### Creating



    create_model -c model_name.json

If `model_name.json` does not exist a new .json file will be created with default
values that can be edited see Model settings.

### Attach Data Sources

To train a model, data sources need to be added. 

    $ attach_data_sources -c model_name.json
    
From there a menu will prompt creating a data source. If the data is setup as
per the example above, then we would.

> Using TensorFlow backend.  
> enter type:   
> 1. paired directory  
> 2. weighted directories  
> 3. remove existing sources.  
> <anything else>. stop  
>...

Enter 1. for paired directories.

> enter images directory: 

Type images *tab complete should work*

> enter labels directory:

Type labels folder name.

> enter a list of angles, leave blank if none

Leave blank

> labellers:  
>  -  0 :  region labels  
>  -  1 :  multiclass labels  
>  -  2 :  categorical labels  
>  -  3 :  double membrane labels  
>  -  4 :  linear labels  
>select labeller:  

Select 1. 

Then you'll be back to the original menu. Press enter and save when prompted to
save. Data sources can be checked with,

    $ inspect_data_sources -c default.json

That will open a matplotlib plot with data sources.


### Training

Once datasources have been added the model can be trained.

    train_model -c model_name.json

This will train the image and at each epoch, model_name-latest.h5 will be written
with the most recent trained results. model_name-best.h5 will be written when
the *best* results have been acheived. 

### Predictions

To predict an image: 

    predict_image model_name.h5 image.tif prediction.tif
    
The model will predict the image and the output will be stored in prediction.tif

## Log Output

**Debugging information**

Example output from depth 3 cascade model.

`#batch_number	batch	cascade_0_acc	cascade_0_binary_accuracy	cascade_0_loss	cascade_1_acc	cascade_1_binary_accuracy	cascade_1_loss	cascade_2_acc	cascade_2_binary_accuracy	cascade_2_loss	loss	size`

Rule: accuracy X, binary accuracy X, loss X

So the index will follow sets of three starting at 2 (element 3 for 1 based 
such as gnuplot. ) 2 + 3*X (3 + 3X for gnuplot)

Example output from depth 3 cascade model.

`#epoch	cascade_0_acc	cascade_0_binary_accuracy	cascade_0_loss	cascade_1_acc	cascade_1_binary_accuracy	cascade_1_loss	
cascade_2_acc	cascade_2_binary_accuracy	cascade_2_loss	loss	val_cascade_0_acc	val_cascade_0_binary_accuracy	val_casc
ade_0_loss	val_cascade_1_acc	val_cascade_1_binary_accuracy	val_cascade_1_loss	val_cascade_2_acc	val_cascade_2_bi
nary_accuracy	val_cascade_2_loss	val_loss`

Starts at element 1 each output has 3 elements (acc, bin acc, los) then it switches
to validation data, with the same indexing. Same organization and order.

## Slurm Scripting

### Predict an image.
For running a script in slurm.

    #!/bin/bash
    #SBATCH --job-name=unet-prediction
    #SBATCH --ntasks=1
    #SBATCH --time=1:00:00
    #SBATCH --mem=64g
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:4
    
    ml Anaconda3/2019.07
    
    unet_env=unet-cuda-conda
    export GPUS=4
    
    echo "#    started: $(date)"
    "$unet_env"/bin/predict_image model_name.h5 image_name.tif prediction_name.tif -b
    echo "#   finished: $(date)"

Then the script can be started with

    sbatch ./testing.sh

For a more complete script:

    #!/bin/bash
    # Simple SLURM sbatch example
    #SBATCH --job-name=unet-prediction
    #SBATCH --ntasks=1
    #SBATCH --time=1:00:00
    #SBATCH --mem=64g
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:4
    home_dir=$(realpath ~)
    image=$2
    image_file=${image/*"/"/}
    working_directory=${image%$image_file}
    model=$1
    model_name="${1/*"/"/}"
    model_name="${model_name/%.h5}"
    
    ml Anaconda3/2019.07
    
    unet_env="$home_dir"/unet-cuda/unet-conda
    
    export GPUS=4
    
    echo "#         start: $(date)"
    echo "#       working:  $working_directory"
    echo "#    model file:  $model_name :: $1"
    echo "#    to predict: $image_file"
    
    echo called with: $1 $2 pred-"$model_name"-"$image_file" -D "batch size=128" "${@:3}" -b
    echo "pwd: " $(pwd)
    "$unet_env"/bin/predict_image $1 $2 pred-"$model_name"-"$image_file" -D "batch size=128" "${@:3}" -b
    echo "#      finished: $(date)"

Now this one can be submitted with additional arguments to specify the model/image.

    sbatch prediction_job.sh model_name.h5 image_name.tif


    
## Model settings

**Outdated**

The model settings 

    "input shape" : (1,64, 64, 64)
 
(channels, slices, height, width) This needs to be smaller than the dimensions of the image being analyzed.
 
     "kernel shape" : (3, 3, 3),

(For convolutions. Each convolution block is by applying this kernel 2x s)
     
     "pooling": (2, 2, 2),
            
(Max pooling layers after convolution blocks. If using fewer z-slices possibly don't pool in z.)
            
     "nlabels" : 2,
    
(number of labels in the output.)

     "depth": 4,

(number of convolution blocks.)

     "model file" : "default-model.h5" 
