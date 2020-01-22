# Training Deeplab on Your Own Dataset



**TLDR**: This tutorial covers how to set up Deeplab within Tensorflow to train your own machine learning model, with a focus on separating humans from the background of a photograph in order to perform background replacement. 

If you'd rather watch this on Youtube, see the [deeplab training tutorial here](https://youtu.be/3NMNP_1d1s8), and the [openCV visualization / background swapping tutorial here](https://youtu.be/4MmFuYoDySQ)

There are 3 parts to the tutorial. Feel free to skip to the section that is most relevant to you.

* Part 1 focuses on collecting a dataset. 
* Part 2 focuses on training Deeplab on your dataset
* Part 3 focuses on visualizing the results of the training, and performing background replacement using openCV.



---



### Installation Process



**Create a Python3 Environment with Pip** 



* With anaconda navigator (or conda) / pyenv / virtualenv, create an environment with python 3.7.4, and activate it:



```
/Users/[username]/.anaconda/navigator/a.tool ; exit;
```



* Anaconda has pip preinstalled. If not using anaconda, make sure you have pip installed. [Follow these directions on a mac](https://www.makeuseof.com/tag/install-pip-for-python/).



**Clone the Deeplab Models Github Repo**

[Clone the official tensorflow models repo](https://github.com/tensorflow/models) 

You will only need the `models/research/deeplab` and `models/research/slim` directories. You can delete everything else.

**Merge the files from the tutorial repo into the tensorflow models repo**

[Clone or download this repo](https://github.com/heaversm/deeplab-training), and put everything into the directory you just created for the tensorflow models repo. but don't overwrite anything *except the `input_preprocess.py` file in the `/deeplab/` directory, which has a small change. 

For example put `models/research/eval-pqr.sh` into the tensorflow `models/research` directory.



**Install Tensorflow**

* From the `models/research/` directory, install tensorflow:



```bash
pip3 install --upgrade pip #need version 19 or higher
```

```bash
pip3 install tensorflow==1.15 #I had issues with tensorflow 2 on a mac
```

If you have a CUDA-compatible GPU, You can use `tensorflow-gpu` instead of tensorflow.



* Install Pillow - this library helps you process images (Python Image Library)

```bash
pip3 install Pillow #use this for a mac. Other systems or versions of python might use PIL
```

* Install other dependencies:

```bash
pip3 install tqdm numpy
```



[more help on installing tensorflow here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md).

Make sure to follow the steps in the link to ensure that you can run `model_test.py`:

```bash
python3 deeplab/model_test.py
```



Pay special attention to this step:



```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

**This command has to be run each time you activate your python environment or open the terminal window**: 

And also, make sure, especially if you are running multiple python environments, that you always use `python3` and `pip3` for every command you run (instead of `python` and `pip`). This will save you *lots* of headaches.



### Image Preparation Process

**Notes**

* Run all `python` commands with `python3`

* Run all `pip` commands with `pip3`

* Must run :

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

for each terminal session





**Dependencies**

Python V3.7.4

Tensorflow 1.15





**Making a dataset**

You will need a consistent background image, and a large set of transparent (or masked) foreground images with photos of people. You'll want to composite each foreground image on to the background.

Make sure the background image is representative of the background image you will be using for real time photo replacement.

Make sure the foreground images represent the diversity of photos you will likely expect in a live scenario. For best results, consider things like:

* Proximity to camera
* Number of people in photos
* Race and gender
* Clothing styles (loose or tight, patterned, dark, light)
* Proximity to each other (touching, hugging, far apart both depthwise and horizontally)
* Poses (sideways, front facing, smiling, acting, etc)
* Props (people holding things, wearing hats or masks, etc)
* Lighting conditions (e.g. high contrast or shadowy, multiple light sources, indoor, outdoor)



**Scraping Images**

The `utilities/scrapeImages.py` file is useful in downloading images from google. 

**NOTE:** this search does not limit search results to freely licensed files - it was only used for my internal testing, and you should be careful not to utilize any scraped images from any website without ensuring that you are adhering to their licensing and use guidelines. 



You should first edit the scrapeImages.py file to use your desired query string. Look for:

```python
url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch&tbs=isz:m,itp:photo,ic:trans,ift:png"
```

The `tbs=` param in this case does the following:

* `ism:m` downloads medium sized images
* `itp:photo` - downloads only photo type images
* `ic:trans` downloads only images with transparency
* `ift:png` downloads only images of file type transparency.



To use your own parameters, do an advanced google search for the type of images you want, and take a look at the query string in the URL bar of your browser for what `tbs` parameters it generates for you, and replace them here.



You then run the scraper as follows:

```bash
python3 utilities/scrapeImages.py --search "[your_search_term]" --num_images 100 --directory "/[Path]/[to]/[your]/[image]/[folder]"
```

Changing `[your_search_term]` and the value of the `--directory` flag to where you want to save images to. 





**Creating Segmentation Images**

You will need to create a new set of images that merges each transparent foreground images on to the consistent background. 

You will also need to create a new set of images where the background is black, and the transparent foreground image matches the color you are trying to segment, in this case "Person" which is color `rgb(192,128,128)`.  

Both sets of images, the "regular" and "segmentation" images should have the same size, and match each other exactly in terms of the position and scale of the foreground subjects in relation to the background. See this example:

**Regular Image**

![athletes-001](https://prototypes.mikeheavers.com/transfer/deeplab/readme_images/athletes-001.jpg)

**Segmentation Image**

![athletes-001](https://prototypes.mikeheavers.com/transfer/deeplab/readme_images/athletes-001.png)

The `photoshop actions` section below has a set of useful actions for accomplishing this properly in photoshop. 

You should make a directory within the `models/deeplab/datasets` directory. Call it whatever you like (in this case, we used `PQR`).

Within that folder, make another folder called `JPEGImages` and place all the "regular" images.



**Photoshop Actions**

If you know how to use photoshop actions, this repo contains a set of actions that will help convert and merge your photos. Go to `window > actions ` in photoshop and choose `load actions` and load the `glowbox.atn` file.

To run these actions in batch, you'll want to go to `File > Automate > Batch` in photoshop, and select the desired action, the folder location of your foreground images. Destination should be `None` as the action contains a save command itself.

To edit any of the actions, you'll want to select the step of the action from the actions panel and double click it, modifying the desired parameters.

* `place_and_save`: 

This action:

*  takes a loaded image
*  resizes the canvas to the size of your background image and places that image as a layer
*  moves your foreground image to the bottom center of the photo
*  exports the image as a 60% quality jpeg.



Make sure to edit the action to specify the location of your background image, the canvas size matching your background image's size, and the desired export quality. Also, make sure the export location does not overwrite your transparent foreground images, you'll need those to create your segmentation masks.



* `segment` : 

This action:

* Takes the transparent images and resizes the canvas to match the dimensions of your chosen background image.
* Aligns the transparent foreground image to the bottom center of the canvas.
* Makes a background and fills it with black (the segmentation color for `background` in deeplab)
* Makes a selection of the foreground and fills it with the proper color for the `Person` segmentation in deeplab: `RGB(192,128,128)` 
* Exports the image as a 60% quality jpeg.

Make sure to edit the action to specify the desired color segmentation for your images, if you are not trying to identify people in your photos. You can see the deeplab (resnet) [color segmentation scheme here](https://github.com/DrSleep/tensorflow-deeplab-resnet/blob/master/images/colour_scheme.png).



* `convert_to_indexed` - You do not have to run this action if you used the `segment` action above. 

However, if you already have images, this action just ensures that the color for the segmentation mask is exact, forcing a `pink-ish` color to the exact pixel values. Photoshop, for example, does some adjustment of colors on a normal save to match your screen's color profile.  You can prevent having to run this action at all if saving from photoshop by ensuring that the `convert to sRGB` option in the `save for web` dialog is unchecked.



* `merge_segmentation` - this action does not need to be run until after all of your model training and image generation has been done. Essentially it is the last step, helping you to visualize how well your machine-learning generated masks actually mask off your subject. It does the following:

* Adds both the regular image and the segmentation masks as layers
* Selects the color range matching the segmentation layer
* Makes a mask around the regular image

You end up with 3 layers - one with the untouched photo, one with the segmentation mask, and one with your regular image masked off to show how well the background was removed and the subjects were isolated.



**Convert your RGB segmentation images to indexed colors**

In order to reduce the number of dimensions of processing deeplab has to do on each image, we will be converting each found RGB color in the segmentation images you made (i.e. `RGB(192,128,128)`) to an indexed color value (i.e. `1`). This will make processing a lot faster.

This repo includes a file in the `deeplab/datasets/` directory called `convert_rgb_to_index.py` which will help you accomplish that. 

Before running, make sure to edit the following: 

```
# palette (color map) describes the (R, G, B): Label pair
palette = {(0,   0,   0) : 0 , #background
   (192,  128, 128) : 1 #person
   }
```

If you are not processing people, the palette should contain all of the segmentation colors you are trying to detect. In our case, since we are just looking for people, the palette contains black for the background as index 0, and pink for the foreground as index 1.

`label_dir`: this is the path (relative to the `datasets` directory where this file is contained) where your Segmentation Class images were saved. Make sure to change it if your file locations differ.

`new_label_dir`: this is the path where your newly generated images will be saved. You do not need to make this directory, it will be generated for you.

To run the script, from the `datasets` directory, run: `python3 convert_rgb_to_index.py`.  You will need to make sure all of this files dependencies are installed via pip:

* `pip3 install Pillow tqdm numpy`



Once it runs, you should have a new folder `SegmentationClassRaw` (or whatever you called the `new_label_dir` folder). It should contain a list of `.png` images. **They will all look black**. This is normal. We converted the RGB values into single index values, so a standard image viewer won't understand this format.







**Make a list of all your training and test images**

Make another folder at the same level as `JPEGImages` called `SegmentationClass` (see the `folder structure` section below for the a better sense of the entire folder structure you will be adding to deeplab). This folder will contain all your segmentation images.

Deciding how to divide up your train and validation set is up to you. Ideally you have at least 500 training images, and at least 100 test images. A good starting split might be a 10:1 ratio of training to test images.



**Generate the tfrecord folder**

Tensorflow has a `tfrecord` format that makes storing training data much more efficient. We will need to generate this folder for our dataset. To do so, this repo has made a copy of the `build_voc2012_data.py` file which has been saved as a new file, (in our case `build_pqr_data.py`). 

Edit the `build_pqr_data.py` file, and make sure there is a flag for our model's desired folders. In this case, look at ~line80:

```python
tf.app.flags.DEFINE_string('image_folder',
                     './PQR/JPEGImages',
                     'Folder containing images.')

tf.app.flags.DEFINE_string(
'semantic_segmentation_folder',
'./PQR/SegmentationClassRaw',
'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
'list_folder',
'./PQR/ImageSets',
'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
'output_dir',
'./PQR/tfrecord',
'Path to save converted SSTable of TensorFlow examples.')
```



Make sure to change any of those directories to match where your files are located. In this instance, the `tfrecord` folder **should** exist. The script will not make it for you. Also note that at around Line 119 I have hardcoded the input format to be `.jpg:

```python
image_filename = os.path.join(
#MH:
#FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
FLAGS.image_folder, filenames[i] + '.jpg')
#END MH
```

and the output images to be `.png`

```python
#MH:
      #filenames[i] + '.' + FLAGS.label_format)
      filenames[i] + '.png')
      #END MH
```

due to an issue I had with the script utilizing the `label_format` flag. You should change those extensions to match the extensions of your own images if they differ.



Now you can run the file (from the `datasets` directory:

```bash
python3 build_pqr_data.py
```



Once this is done, you will have a `tfrecord` directory filled with `.tfrecord` files.



**Add the information about your dataset segmentation** (TODO: check to make sure we still need this step...)



You'll need to provide tensorflow the list of how your dataset was divided up into training and test images. 

In `deprecated/segmentation_dataset.py` , look for the following (~Line 114):

```
# MH
_PQR_INFORMATION = DatasetDescriptor(
splits_to_sizes={
  'train': 487,
  'val': 101,
  'trainval': 588,
},
num_classes=2,
ignore_label=255,
)

_DATASETS_INFORMATION = {
'cityscapes': _CITYSCAPES_INFORMATION,
'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
'ade20k': _ADE20K_INFORMATION,
'pqr': _PQR_INFORMATION,
}
# END MH
```

These splits should match the number of files in your training and test sets that you made earlier. For example, if `train.txt` has 487 line numbers, `train` is 487. Same with `val` and `trainval`. If you are trying to segment more than just the background and foreground, `num_classes` should match the number of segmentations you are targeting. `ignore_label=255` just means you are ignoring anything in the segmentation that is white (used in some segmentations to create a clear space division between multiple segmentations).



Note that `_DATASETS_INFORMATION` also contains a reference to this new dataset descriptor we've added:

`'pqr': _PQR_INFORMATION`



You're finally ready to train!



### Training Process



**Folder Structure**

Make sure your folder structure from `/datasets` looks similar to this, if you followed all of the naming conventions in the above steps:


```
+ PQR
  + exp //contains exported files
  + train_on_trainval_set
  + eval //contains results of training evaluation
  + init_models //contains the deeplab pascal training set, which you need to download
  + train //contains training ckpt files
  + vis
    + segmentation_results //contains the generated segmentation masks
  + Imagesets
    train.txt
    trainval.txt
    val.txt
  + logs
  + tfrecord //holds your converted dataset
buid_pqr_data.py //creates your tfrecord files
convert_rgb_to_index.py //turns rgb images into their segmentation indices

../../train-pqr.sh //holds the training script
../../eval-pqr.sh //holds the eval script
../../vis-pqr.sh //holds the visualization script
```



**Download the Pascal Training Set**

In order to make our training *much* faster we'll want to use a pre-trained model, in this case pascal VOC2012. [You can download it here](http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz). Extract it into the `PQR/exp/train_on_tranval_set/init_models` directory (should be named `deeplabv3_pascal_train_aug`).









**Edit your training script**

First, edit your `train-pqr.sh` script (in the `models/research`) directory:

```bash
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
PQR_FOLDER="PQR"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
DATASET="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/tfrecord"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"

NUM_ITERATIONS=9000
python3 "${WORK_DIR}"/train.py \
--logtostderr \
--train_split="train" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--train_crop_size=1000,667 \
--train_batch_size=4 \
--training_number_of_steps="${NUM_ITERATIONS}" \
--fine_tune_batch_norm=true \
--tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
--train_logdir="${TRAIN_LOGDIR}" \
--dataset_dir="${DATASET}"
```

Things you may want to change:

* Make sure all paths are correct (starting from th `models/research` folder as `CURRENT_DIR`)
* `NUM_ITERATIONS` - this is how long you want to train for. For me, on a Macbook Pro without GPU support, it took about 12 hours just to run 1000 iterations. You can expect GPU support to speed that up about 10X. At 1000 iterations, I still had a loss of about `.17`. I would recommend at least 3000 iterations. Some models can be as high as about 20000. You don't want to overtrain, but you're better off over-training than under-training.
* `train_cropsize` - this is the size of the images you are training on. Your training will go **much** faster on smaller images. 1000x667 is quite large and I'd have done better to reduce that size a bit before training. Also, you should make sure these dimensions match in all three scripts: `train-pqr`,`eval-pqr`, and `vis-pqr.py`.
* The checkpoint files (`.ckpt`) are stored in your `PQR_FOLDER` and can be quite large (mine were 330 MB per file). However, periodically (in this case every 4 checkpoint files), the oldest checkpoint file will be deleted and the new one added - this should keep your harddrive from filling up too much. But in general, make sure you have plenty of harddrive space.







**Start training**:

You are finally ready to start training!

From the `models/research` directory, run `sh train-pqr.sh`

If you've set everything up properly, your machine should start training! This will take.a.long.time. You should be seeing something like this in your terminal:



![training](https://prototypes.mikeheavers.com/transfer/deeplab/readme_images/training.png)



**Evaluation**

Running `eval-pqr.sh` from the same directory will calculate the [`mean intersection over union`](https://www.jeremyjordan.me/evaluating-image-segmentation-models/) score for your model. Essentially, this will tell you the number of pixels in common between the actual mask and the prediction of your model:

![iou](https://prototypes.mikeheavers.com/transfer/deeplab/readme_images/iou.jpg)



In my case, I got a score of ~`.87` - which means essentially 87% of the pixels in my prediction mask were found in my target mask. The higher the number here, the better the mask.



**Visualization**

To visualize the actual output of your masks, run `vis-pqr.sh` from the `models/research` directory. These will output to your visualization directory you specified (in our case, `models/research/deeplab/datasets/PQR/exp/train_on_trainval_set/vis/segmentation_results`).  You will see two separate images for each visualization: the "regular" image, and the "prediction" (or segmentation mask). 

If you want to combine these two images, the `merge_segmentation` photoshop action can help. 

I've also set this up as an automated process in openCV to take an image and its segmentation mask and automatically substitute in a background of your choosing.





### Using OpenCV for background replacement



**Install OpenCV**

[Follow these directions to install opencv on mac](https://www.pyimagesearch.com/2018/08/17/install-opencv-4-on-macos/) - but use version 4.1.2 instead of 4.0:

```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.2.zip
```

Give your virtual environment a name of cv, then `workon cv`.

Rename` /usr/local/lib/python3.7/site-packages/cv2/python-3.7/cv2.cpython-37m-darwin.so` to `cv2.so`

then `cd /Users/[your_username]/.virtualenvs/cv/lib/python3.7/site-packages`

then `ln -s /usr/local/lib/python3.7/site-packages/cv2/python-3.7/cv2.so cv2.so`



The cv Python virtual environment is ***entirely independent and sequestered*** from the default Python version on your system. Any Python packages in the *global directories* *will not* be available to the cv virtual environment. Similarly, any Python packages installed in site-packages of cv *will not* be available to the global install of Python.



**Directory Structure**



Navigate to the `cv` directory. You should have the following directory structure:

```
+input 
+output
+masks
+bg
replacebg_dd.py
```



* `/input`  - contains the images whose background you want to replace
* `/masks` - contains the segmentation masks that will separate the foreground from the background (people from everything else).
* `/output` - where the photos with the replaced background will be saved
* `/bg` contains the background image that will be used as the replacement.
* `replacebg_dd.py` - the python script that utilizes opencv to handle background replacement.



**Note**:  all files in the input and masks directories should have the same names to ensure they match up together when running the script



**Using the replacebg.py script**:



Before calling the script, check the following lines within the script:

```
input_dir = 'input/'
output_dir = 'output/'
mask_dir = 'masks/'
bg_dir = 'bg/'
bg_file = 'track.jpg'
```

These directories should match your directories relative to the `replacebg.py` script.

`initial_threshold_val = 150` : Changing this value will change the black / white value above which the foreground is kept rather than the background.





**Script Options**

The python script is responsible for handling what pixels to keep from the source vs which to throw away, and can do some basic thresholding and blurring of the mask image to attempt to improve results.

There are a few parameters you can pass the `replacebg.py` script:

* `--image` (i.e. `replacebg.py --image 36`) would show (but not save) the image numbered 36
* `--generate` (i.e. `replacebg.py --generate 20`) would save out the first 20 images
* `--all` (`replacebg.py --all`) would save out all images (provided you manually keep the `num_inputs` variable synched with however many files you have in your input directory)
* `replacebg.py --start 20` would generate images between the 20th and `num_inputs` photos.
* `replacebg.py --start 20 --end 30` would generate images between the 20th and 30th photos in the directory



**Keyboard commands**

When you run the script and it is displaying an image, you can use the following keyboard commands:



* `z` increases the threshold, tightening up on the subjects and revealing more of the substituted background
* `x` decreases the threshold, showing more of the source photo
* `s` saves the image out
* `q` quits the window and script execution
* `i` cycles to the next image in the sequence







---




**NOTE:**

This tutorial and repo were created through my difficulties installing and training deeplab, in the hopes that it would make things easier for others trying to do the same. Very little of the code is my own, and has been assembled from a variety of sources - all of which were extremely helpful, but none of which I was able to follow on their own in order to successfully train Deeplab. By combining various pieces of the following links, I was able to create a process that worked smoothly for me. 



**Links**:

[Analytics Vidhya](https://www.analyticsvidhya.com/blog/2019/02/tutorial-semantic-segmentation-google-deeplab/#comment-157463) - *Semantic Segmentation: Introduction to the Deep Learning Technique Behind Google Pixel’s Camera!,* Saurabh Pal

[Installing Tensorflow](https://www.tensorflow.org/install/) - Official Documentation

[Installing Deeplab](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md) - Official Documentation

[Tensorflow-Deeplab-Resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) - Dr. Sleep

[Free Code Camp](https://medium.com/free-code-camp/how-to-use-deeplab-in-tensorflow-for-object-segmentation-using-deep-learning-a5777290ab6b) - *How to use DeepLab in TensorFlow for object segmentation using Deep Learning*, Beeren Sahu

[Dataset Utils](https://github.com/ml4a/ml4a-guides/tree/master/utils) - Gene Kogan - useful in scraping images for a dataset and creating randomly sized, scaled, and flipped images in order to increase the training set size.


