# Deep Bilateral Network
Deep bilateral network based on the paper [here](https://groups.csail.mit.edu/graphics/hdrnet/). This network predicts local transformations from a low resolution input and applies them to a high resolution input in an adaptive way using a bilateral slicing layer. It can be thought of as a network that learns a black-box image filter. The main advantages of this network are:

- Speed. Very shallow network, and bulk of computation done at low resolution.
- High resolution output.
- Edge-preserving since it learns transforms rather than output spaces directly. As such, it is less susceptible to artifacts.

### Requirements
- Python >= 3.6
- PyTorch >= 0.4
- tensorboardX
- openCV (for generating videos)
- scikit-video (for generating videos)
- CUDA (preferably, but not necessary)

### Project structure
The code is organized primarily into the following files and directories:

- `configuration.py`: Location to specify arguments with which to train/evaluate/run. Global variables in this file are modified instead of specifying script command line parameters.
- `train.py`: Script used to train on a dataset. Acquires all parameters from `configuration.py`.
- `eval.py`: Script used to evaluate on a dataset. Acquires all parameters from `configuration.py`.
- `generate_videos.py`: Script used to run a trained model on a directory of videos and get the output. Acquires parameters from `configuration.py`, and can specify the input/output video directories directly within the script.
- `models/`: Contains the main `DeepBilateralNetCurves` model and its subclasses that use slightly different modules.
- `datasets/`: Contains the input pipeline classes. `BaseDataset.py` defines a dataset that works for general purposes, but other subclasses can be created using it as a model. If you create a new dataset class, make sure to add the corresponding code needed to specify it in `configuration.py`.
- `saved_runs/`: Contains the saved models and output logs from training.
- `bilateral_slice_op/`: Contains the C++ code for the custom bilateral slice layer as well as the code needed to build it.

### Building the bilateral slice layer
This model uses a custom layer with C++ implementation that must be built prior to training/running the model.

1. `cd bilateral_slice_op`
2. `python setup.py install`

### Training
1. Create a dataset. If using the `datasets/BaseDataset.py` loader, see `data/debug` for an example of how to structure the data. Note that you can quickly run the model on this dataset just to make sure that everything is working and that you have successfully built the bilateral slice layer.
2. Specify the model and training parameters in `configuration.py` by modifying the global variables defined near the top of the file.
3. `python train.py [run_name]`
4. For viewing the learning curve and all evaluation results using TensorBoard, run `tensorboard --logdir=saved_model/[run_name]` and open the port to see the evaluation results. If don't know how to use tensorboard, can check  [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard).

### Evaluating
1. Specify the model parameters in `configuration.py` by modifying the global variables defined near the top of the file.
2. `python eval.py`

### Generating videos
1. Specify the model parameters in `configuration.py` by modifying the global variables defined near the top of the file. Note that the variable `pretrained_path` must specify the path to a `.pth` model file.
2. Specify the `input_dir` and `output_dir` global variables at the top of the `generate_videos.py` file.
3. `python generate_videos.py`

### Conversion

##### CoreML
1. Pull the following [conversion repository](https://bitbucket.org/modiface_inc/pytorch_to_coreml/src/master/) (proprietary).
2. Copy your trained `model.pth` file to `deep_bilateral_network/` in the conversion repository.
3. Modify model parameters as well as input and output shapes inside the `load_model()` function of the `deep_bilateral_network/convert.py` file of the conversion repository.
4. Follow the conversion repository instructions for running the conversion script.

\* Notes:

- The converted model does not include the final output following the bilateral slice layer at this time. This is because we have yet to write a custom CoreML layer. Instead, both the coefficients and guidemap are returned.
- The implementation for a linear combination of ReLU's applied when computing the guidemap is currently hacked together using a series of other layers. Writing a custom CoreML layer for this would probably improve performance.

##### NCNN:
todo: Add documentation for this conversion