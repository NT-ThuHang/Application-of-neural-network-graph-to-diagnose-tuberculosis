# Application-of-neural-network-graph-to-diagnose-tuberculosis
Download data at [kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset), [NIH](https://tbportals.niaid.nih.gov/download-data)

Preprocess clahe
*   `python3 source/transform_clahe.py sample_data`
 
Train from directory
*   `python3 source/main.py sample_data config.ini`

Train from processed dataset
*   `python3 source/main.py sample_data.pt config.ini`

Predict a single image 
*   `python3 source/predict.py model.pt sample_data/Normal/Normal-1.png`
