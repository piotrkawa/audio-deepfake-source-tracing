# Baselines for Interspeech 2025 Special Session on Source Tracing 

The following repository contains baselines to start your work with the task of DeepFake Source Tracing as 
part of [Source tracing: The origins of synthetic or manipulated speech](https://www.interspeech2025.org/special-sessions) INTERSPEECH 2025 Special Session.

## Attribution

Special thanks to [Resemble AI](https://www.resemble.ai) and [AI4Trust project](https://ai4trust.eu/) for their support and affiliation.

## Contributors
 - [Piotr Kawa](https://github.com/piotrkawa),
 - [Adriana Stan](https://github.com/adrianastan),
 - [Nicolas M. Müller](https://github.com/mueller91).


## Before you start

### Download dataset

The baseline is based on the [MLAAD (Source Tracing Protocols) dataset](https://deepfake-total.com/sourcetracing).
To download the required resources run:
```bash
python scripts/download_resources.py
```

The default scripts' arguments assume that all the required data is put into `data` dir in the project root directory.

### Install dependencies

Install all the required dependencies from the `requirements.txt` file. The baseline was created using Python 3.11.
```bash
pip install -r requirements.txt
```


### Evaluation metrics

Coming soon.


## GE2E + Wav2Vec2.0 Baseline

To train the feature extractor based on Wav2Vec2.0-based encoder using [GE2E-Loss](https://arxiv.org/pdf/1710.10467) run:
```python
python train_ge2e.py --config configs/config_ge2e.yaml
```


## REFD Baseline

This baseline builds upon the work of Xie et al. ["Generalized Source Tracing: Detecting Novel Audio Deepfake Algorithm 
with Real Emphasis and Fake Dispersion Strategy"](https://arxiv.org/abs/2406.03240) and its associated [Github repo](https://github.com/xieyuankun/REFD/).

The work uses a data augmentation technique and an OOD detection method to improve the classification of unseen
deepfake algorithms. However, in this repository we implement the very basic setup, and leave potential 
authors the option to improve upon it. 


<details>
  <summary>More details here</summary>


### Download data augmentation datasets


For the required data augmentation step you will need the [MUSAN](https://www.openslr.org/17/) and [RIRS_NOISES](https://www.openslr.org/28/) datasets.



### Step 1. Data augmentation and feature extraction

The first step of the tool reads the original MLAAD data, augments it with random noise and RIR and extracts
the `wav2vec2-base` features needed to train the AASIST model.  Additional parameters can be set from the script,
such as max length, model, etc. 

```bash
python scripts/preprocess_dataset.py
```

Output will be written to `exp/preprocess_wav2vec2-base/`. You can change the path in the script. 

### Step 2. Train a AASIST model on top of the wav2vec2-base features

Using the augmented features, we then train an AASIST model for 30 epochs. The model is able to classify the samples
with respect to the source system. The class assignment will be written to `exp/label_assignment.txt`.

```bash
python train_refd.py
```

### Step 3. Get the classification metrics for the known (in-domain) classes

Given the trained model stored in `exp/trained_models/`, we can now compute its accuracy over known classes (those
seen during training time).

```bash
python scripts/get_classification_metrics.py
```

The script will limit the data in the `dev` and `eval` sets to the samples which are from the known systems 
(i.e. those also present in the training data) and compute their classification metrics.

### Step 4. Run the OOD detector and evaluate it

```bash
python scripts/ood_detector.py --feature_extraction_step
```
The script builds an NSD OOD detector as described in the original paper. The OOD detector is based on the hidden states and logits of the AASIST model. It first extracts all this info from the trained model and stores it in separate dicts. It then loads the training data and determines the in-domain scores. 

It then computes the scores for the development set. Based on these scores for which we know the OOD class assignments
it determines the EER and associated threshold. The computed threshold is then used for providing the 
classification into OOD and known systems metrics for the evaluation data. 

The baseline results at this point is a 63% EER with an F1-score of 0.31 for the eval data. 

</details>


## License
This repository is licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/) for original content.


### Exceptions:
- Portions of this repository include code from [REFD repository](https://github.com/xieyuankun/REFD/), which does not have a license.
- As per copyright law, such code is "All Rights Reserved" and is not covered by the CC BY-NC license. Users should not reuse or redistribute it without the original author's explicit permission.


## References
The following repository is built using the following open-source repositories:
* [coqui.ai TTS](https://github.com/coqui-ai/TTS),
* [REFD](https://github.com/xieyuankun/REFD).

