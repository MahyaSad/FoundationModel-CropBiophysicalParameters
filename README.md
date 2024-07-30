# Estimating Crop Biophysical Parameters through Self-Supervised Learning with Foundation Models and Multimodal SAR and Optical Observations

ecent advancements in both remotely sensed datasets and machine learning/deep learning techniques have made these tools popular for various agricultural applications, such as crop classification, monitoring, and yield prediction. However, the scarcity of labeled data for crop monitoring and biophysical parameter estimation has made it challenging to use deep learning techniques effectively.

Geospatial foundation models with self-supervised learning (SSL) using large unlabeled datasets can be fine-tuned with small labeled datasets for different downstream tasks, addressing this limitation. In our paper, we utilize SSL with Masked Autoencoders (MAE) and Vision Transformers (VIT) for estimating crop Volumetric Water Content (VWC) and height. The SSL model focuses on spatial modeling, while the supervised learning model incorporates both temporal and spatial dimensions.

Insert images here

Citation
If you use this code, please cite our paper:

bibtex
Copy code
@article{
}


# Methodolody Overview



Our approach involves several key components and methodologies:

# Self-Supervised Learning (SSL)
For running the SSL part for SAR and Optical data individually, refer to the Python scripts in the SSL subfolder. You can also find the pre-trained models for both sensors in this subfolder.

# Supervised Learning (MTL and STL)
To run the supervised learning models for Multi-Task Learning (MTL) or Single-Task Learning (STL), refer to the supervised_learning subfolder. You will need the pre-trained SSL model to run these models.

# Random Forest and XGBoost
For running Random Forest and XGBoost models, look for the VWC and height estimation Python codes in the ML subfolder.

Insert images here
