You are a senior professor, professor in software engineering and I need a complete solution for an image classification assignment (CSCD618/DSCD604 Assignment 2). 

## ASSIGNMENT REQUIREMENTS:

1. **Dataset Structure**: 
   - I have a dataset with train directory and test directory containing images
   - 16 categories/classes to classify
   - class_names.csv file listing the 16 classes
   - sample_submission.csv with test image filenames

2. **Model Architecture**:
   - Build a custom MobileNet-like architecture with < 4.0M parameters
   - Must extract discriminative features from images
   - Design for top-1 classification accuracy on 16 categories

3. **Training Requirements**:
   - Use advanced data augmentation: random flips, rotations, color jittering, etc.
   - Optimize using SGD with momentum OR Adam
   - Use Cosine annealing warm restarts scheduling
   - Train to achieve high accuracy on validation set

4. **Submission**:
   - Generate predictions for unlabelled test images
   - Output CSV with exactly two columns: ID, Prediction
   - ID must match filenames from sample_submission.csv exactly
   - Prediction must be valid class labels from class_names.csv
   - Max 10 daily submissions allowed

5. **Evaluation**:
   - Classification accuracy: (correct predictions / total test images)
   - Higher accuracy = better performance

6. **Code Submission**:
   - Well-organized code with description of how to run
   - Must include complete pipeline from data preprocessing to prediction

## WHAT I NEED:

Please provide:

1. **Complete Python code** with all necessary imports, functions, and classes
2. **Training script** that:
   - Loads and preprocesses images from train directory
   - Implements the custom MobileNet-like architecture (<4M params)
   - Applies advanced data augmentation
   - Trains with SGD/Adam + Cosine annealing warm restarts
   - Saves the best model checkpoint
   - Logs training/validation metrics

3. **Inference script** that:
   - Loads the saved model
   - Processes test directory images
   - Generates predictions matching class_names.csv
   - Creates submission CSV in exact required format

4. **Data loading code** that:
   - Reads class_names.csv to get class labels
   - Reads sample_submission.csv to get test filenames
   - Properly loads images from train and test directories

5. **Model architecture details**:
   - Should be based on MobileNet principles (depthwise separable convolutions)
   - Parameter count must be < 4,000,000
   - Appropriate for 16-class classification

6. **Hyperparameter recommendations**:
   - Image size (suggest 224x224 or 256x256)
   - Batch size
   - Learning rate and scheduler settings
   - Number of epochs
   - Optimizer choice and settings
   - Data augmentation parameters

7. **Code organization**:
   - Train.py
   - Model.py
   - DataLoader.py
   - Inference.py
   - Utils.py
   - Requirements.txt
   - README.md with instructions

8. **Additional considerations**:
   - Implement early stopping to prevent overfitting
   - Use label smoothing if beneficial
   - Include gradient clipping
   - Implement test-time augmentation (TTA) for better predictions
   - Add mixup or cutmix augmentation for improved generalization
   - Use pretrained weights if allowed (imagenet initialization)

## SPECIFIC TECHNICAL REQUIREMENTS:

- Framework: PyTorch or TensorFlow/Keras (PyTorch preferred)
- GPU support required
- Must handle variable image formats (JPEG)
- Efficient data loading with multiprocessing
- Memory optimization for training

## DESIRED OUTPUT:

Provide the complete codebase that I can run end-to-end. Include all necessary explanations, justifications for design choices, and ensure the code is production-ready for the Kaggle competition environment.

The goal is to achieve the highest possible accuracy on the leaderboard while meeting all assignment constraints (parameter limit, architecture requirements, etc.).

Please provide the full solution now.
BONUS: ADDITIONAL TECHNICAL DETAILS
If you want even more specific guidance, add these details to the prompt:

text
Add these specific technical implementations:

1. **MobileNet-like Architecture**:
   - Use depthwise separable convolutions
   - Include BatchNorm and ReLU after each convolution
   - Global Average Pooling before final FC layer
   - Model should be lightweight but powerful

2. **Data Augmentation**:
   - RandomHorizontalFlip (p=0.5)
   - RandomRotation (degrees=15)
   - ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
   - RandomResizedCrop
   - Normalization with ImageNet stats
   - Optional: AutoAugment or RandAugment

3. **Training Strategy**:
   - Cosine annealing with warm restarts (T_0=10, T_mult=2)
   - Learning rate: start with 0.01 (SGD) or 0.001 (Adam)
   - Weight decay: 1e-4
   - Gradient clipping: max_norm=1.0

4. **Validation Strategy**:
   - Split train data into train/val (80/20 or use provided validation set)
   - Monitor validation accuracy for early stopping
   - Save best model based on validation accuracy

5. **Ensemble/Test-time Augmentation**:
   - Use TTA with 5-10 augmentations per test image
   - Average predictions for final result

6. **Output Format**:
   ```csv
   ID,Prediction
   test_000001.JPEG,class_name
   test_000002.JPEG,class_name
Make sure the code handles:

Missing images gracefully

Proper image resizing

Correct label mapping

Memory-efficient batching

Progress bars for training

GPU/CPU compatibility

text

---

## EXPECTED OUTPUT STRUCTURE

The AI should provide:

### File Structure:
assignment2/
├── train.py
├── inference.py
├── model.py
├── data_loader.py
├── utils.py
├── config.py
├── requirements.txt
├── README.md
└── run.sh

text

### Key Components:
1. **Model architecture with <4M parameters**
2. **Complete training pipeline**
3. **Inference pipeline for submission**
4. **Detailed README with setup instructions**
5. **All hyperparameters tuned for performance**

### Expected Accuracy:
- Aim for >85-90% validation accuracy
- Use ensemble/TTA for final submission
- Target top-10 on Kaggle leaderboard