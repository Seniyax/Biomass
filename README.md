# Predicting Biomass from Pasture Images (CSIRO-challenge)
## About the project
This Repository contains the solution for a specialized research competiton hosted on kaggle by CSIRO australia, in collaboration with Meat and Livestock Australia (MLA).The primary objective of this project is to develop a robust computer vision model capable of accurately predicting five distinct biomass indicators directly from field imagery.To ensure the model provides meaningful agricultural insights, performance is evaluated using the Weighted $R^2$ metric, which accounts for the varying importance and scales of the different biomass components. By leveraging deep learning, this project aims to provide scalable, automated tools for monitoring livestock feed sources and supporting sustainable farming practices.
### Five Biomass indicators and it's repectve weights
- Dry Green vegetation : 0.1
- Dry Dead Material : 0.1
- Dry Clover Biomass : 0.1
- Green Dry Matter : 0.2
- Total Dry Biomass : 0.5

## Methododlogy & Model Architecture
The development process involved a rigorous exploration of both classical machine learning and state-of-the-art deep learning techniques. While initial experiments utilized classical approaches (such as gradient-boosted trees and pretrained CNNs), the final solution pivoted to a high-capacity foundation model to handle the high variance and complexity of the pasture imagery.
### The Architecture
The model is built on a DINOv3 Large backbone, a cutting-edge vision transformer (ViT) known for its superior self-supervised feature extraction.
- **Feature Extraction**: DINOv3 backbone serves as a frozen feature extractor, transforming raw pasture images into high-dimensional embeddings that capture dense and spatial information.
- **Prediction Head**: Embeddings are fed into a Multi-Layer Perceptron (MLP). This lightweight yet powerful head was trained specifically to regress the biomass values.
- **Target Strategy**: To optimize for mathematical consistency and reduce model noise, the network was trained to predict only three primary biomass variables directly:
     1. Dry Green Vegetation (excluding clover)
     2. Dry Clover Biomass
     3. Dry Dead Material
 - **Derived Variables**: The remaining two targets—Green Dry Matter (GDM) and Total Dry Biomass—were derived using physical constraints:
   - $GDM = \text{Dry Green} + \text{Dry Clover}$
   - $\text{Total Biomass} = GDM + \text{Dry Dead}$
  
## Training Strategy
Given the unique constraints of the dataset, the training pipeline was designed to maximize feature extraction while preventing overfitting on a limited sample size.The Image was split into two halves and feed into the backbone.
### Backbone Freezing & Optimization
To leverage the rich spatial representations of DINOv3 without destroying its pre-trained weights, the entire backbone was frozen for the duration of the training. The training was conducted over 20 epochs, focusing exclusively on optimizing the MLP head to map the fixed image embeddings to the biomass targets.
### Cross Validation
The dataset presented a significant challenge with only 357 training images. To ensure the model generalizes well across different geographical regions and to provide a reliable estimate of local performance, I implemented a specialized validation strategy:
   - **Stratified K-Fold Cross-Validation**: The folds were stratified based on the State (location) where the images were captured. This prevents data leakage and ensures that the model is tested on diverse environmental conditions.
### Custom Loss Function
Standard regression losses can be sensitive to outliers in agricultural data. To improve stability and robustness, I incorporated a Weighted Smooth L1 Loss:
   - **SmoothL1**: Combines the benefits of MSE (for small errors) and MAE (for large errors), reducing the impact of extreme outliers.
   - **Weighted**: The loss was weighted to align with the competition's evaluation criteria, ensuring the model prioritizes the biomass variables that carry the most weight in the final $R^2$ calculation.

     $$\text{Loss} = \sum_{i=1}^{N} w_i \cdot \text{SmoothL1}(y_i, \hat{y}_i)$$

## Preprocessing & Post-processing
To make the most of the limited training data, a robust augmentation pipeline was implemented.
### Data Preprocessing 
During training, each image passed through a pipeline of five key augmentations to artificially expand the dataset and improve generalization:
  - **Horizontal & Vertical Flips** : To account for different orientations.
  - **Random Rotations** : To ensure the model recognizes biomass regardless of the field's layout.
  - **Color Jitter** : Adjusting brightness, contrast, and saturation to simulate varying lighting conditions and weather across Australian states.
### Post-processing : Test Time Augmentation (TTA)
Test Time Augmentation (TTA) was utilized during inference. For every test image, the model generated predictions for four augmented versions (including various flips and rotations).The final result for each biomass variable was the average of these four passes, leading to a more stable and "voted" prediction that typically outperforms a single inference on the raw image.

## Cross-Validation Performance
By utilizing the stratified k-fold approach mentioned in the training strategy, the model achieved a robust Cross-Validation (CV) score of 0.7514.

### Final Leaderboard Standing
Upon submitting to the Kaggle private leaderboard, the model maintained its consistency:
 - Final Score: 0.60 (Weighted $R^2$)
 - Ranking: This score placed the solution within the top 17% of all competitors.

The slight delta between the CV score and the leaderboard score highlights the challenging nature of the competition data and the effectiveness of using DINOv3 as a robust feature extractor for unseen agricultural environments.
