# GENEA 2023 baselines
In this repository, we provide two example models for the 2023 GENEA challenge:
1. the first (**monadic**) model is only trained on the main agent's own speech; while
2. the second (**dyadic**) model also considers the interlocutor's speech and movements as its conditioning input. 

The models are based on the [paper](https://openreview.net/forum?id=gMTaia--AB2) *"The IVI Lab entry to the GENEA Challenge 2022 – A Tacotron2 Based Method for Co-Speech Gesture Generation With Locality-Constraint Attention Mechanism"*.

The implementation is kindly provided by [Che-jui Chang](https://sites.google.com/view/chejuichang/) from Rutgers University. If you use this repository in your work, please cite their paper using the following `.bib` entry:

```
@inproceedings{chang2022ivi,
  title={The IVI Lab entry to the GENEA Challenge 2022--A Tacotron2 based method for co-speech gesture generation with locality-constraint attention mechanism},
  author={Chang, Che-Jui and Zhang, Sen and Kapadia, Mubbasir},
  booktitle={Proceedings of the 2022 International Conference on Multimodal Interaction},
  pages={784--789},
  year={2022}
}
```
## Environment
To install the required libraries, activate your python environment and use the following command:
```
pip install -r requirements.txt
```

*Note:
This repository was tested under `Ubuntu 20.04` and `Python 3.9` using the above libraries, but it may still work under different configurations.*

## Data Processing
### Download GENEA23 Data
Get the dataset from the [official GENEA 2023 repository](https://github.com/genea-workshop/genea_challenge_2023/tree/main/dataset) and unzip all files to a folder of your choosing.

### FastText Word Embeddings
Download and unzip the [word embedding](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip) from FastText. Put the file "crawl-300d-2M.vec" under the project directory.

```sh
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
rm crawl-300d-2M.vec.zip
```

### Data Preprocessing
Preprocess the data and save them as h5 files. (This might take a while.) 
```
python process_data.py -d <path_to_dataset> 
```

The above script generates the following four h5 files under the project directory:
* `trn_interloctr_v1.h5`
* `trn_main-agent_v1.h5`
* `val_interloctr_v1.h5`
* `val_main-agent_v1.h5`
---
Calculate audio statistics based on the training set:
```
python calculate_audio_statistics.py
```

### Create Motion Processing Pipelines
We will use the following command to generate pipelines (*.sav) for converting our motion representations to euler rotations.
```
python create_pipeline.py
```


## Train the Model
We provide two baseline configurations: a monadic system (generating gestures from the agent's own speech only), and a dyadic system (which is also conditioned on the interlocutor's behaviour).

The training configurations are defined in `Tacotron2/common/hparams_monadic.py` and `Tacotron2/common/hparams_dyadic.py`. You might want to edit the following options:
1. Enter the `output_directory` and `device` of your own choice. 
2. Add the `checkpoint_path` if you would like to resume training. 
3. Make sure to edit `n_acoustic_feat_dims`, 78 for full-body training, and 57 for upper-body training only. 

Train the model using the following commands:

```
cd Tacotron2
python train_monadic.py # (Option 1)
python train_dyadic.py  # (Option 2)
```

The weights and logs can be found under the `output_directory`. It takes roughly 20k~30k iterations to produce decent results.

## Testing the Model
The below codes generate all gestures given the processed validation data (`val_main-agent_v0.h5`). You can use the same scripts to generate the motion for all test sequences once the test inputs are released near the end of the challenge.

```
python generate_all_gestures.py -ch <checkpoint_path> -t full
```

The bvh files should be generated under "Tacotron2/outputs/" folder. By defaut, the cuda device "0" is used. If you prefer to use a different cuda device for inference, please edit line 23 in the `Tacotron2/common/hparams_monadic.py` or `Tacotron2/common/hparams_dyadic.py` files.

Note: 
You can import the bvh files to Blender to visualize the gesture motions. 
If you would like to render the motions, please reach out to the [repository](https://github.com/TeoNikolov/genea_visualizer) provided by GENEA Challenge 2023. 




