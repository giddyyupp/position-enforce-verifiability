# Example Metric Notebooks

In this folder, we share two example metric calculation notebooks. 

We further conducted a small user study with five experienced reviewers to measure the average time required to apply the verification instructions explained in our paper.

* A short [script](./1_ObjectDetection_Eval.ipynb) which takes around 5 minutes to review. 

* A longer [script](./2_AnomalyDetection_Eval.ipynb) that takes between 45 minutes to 1 hour to review. 


## Google Colab Examples

We deploy these two notebooks on Google Colab to enable straightforward inspection and execution without requiring local environment setup.


1. **COCO detection evaluation notebook.** This notebook evaluates object detection predictions on COCO using `pycocotools` for standard metric computation, and automatically retrieves the required annotation files from the official COCO website.

https://colab.research.google.com/drive/1jPhflV8ImH_7bbtDEKiKJChPTIBG7xl1?usp=sharing


2. **MVTec-AD anomaly detection evaluation notebook.** This notebook evaluates anomaly detection results on the MVTec-AD dataset, where predictions are given as segmentation masks. It automatically fetches the dataset from the official source, and computes the evaluation metrics using a custom implementation built on top of `sklearn.metrics`.

https://colab.research.google.com/drive/1rtpKkQF1kIMZzzTKRfeJmHMiP_JNmrSu?usp=sharing