# Medical-image-classifier
Image Classifier — A ResNet-50 based Streamlit app that classifies images as medical or non-medical. 

This repository contains a complete pipeline for building a binary image classifier (medical vs non-medical)
and a Streamlit app to extract images from PDFs / URLs and classify them.

## Approach & Reasoning
Thought of different models, had a dry run, read and reasearched about different sort of models but I finally ended up using ResNet50. Before it I tried CLIP, Open AI API & ResNet18 but none of them felt satisfactory. CLIP & Open AI API felt very easy nothing to train or tune or anything, ResNet18 had a little more speed at the cost of accuracy but as the task is about classifying medical and non medical images I thought that accuracy should be at the frontfoot instead of speed, so at the end I opted ResNet50. 
1. Where first I gathered multiple images more for medical dataset compared to non-medical dataset. The dataset was big enough, medical dataset had nearly 10k images and non medical also had somewhere around 1k.
2. Used split_dataset.py to split the dataset unto train/val/test, then used it to train the model, after training the model I tested it and generated evaluation results like accuracy, precision, recall, F1 score and Confusion Matrix.
3. Moving further ahead I loaded the model to the streamlit app for the purpose of classifying image from pdf and url's, which are extracted, classified and give you three options first to "Download predictions CSV" (comes with the confidence score), "Download MEDICAL images ZIP" & "Download NON-MEDICAL images ZIP" i.e, from the classified images you can seperate them and install into two different zip files as per their label.

## Accuracy results & Confusion Matrix
  <img width="389" height="249" alt="image" src="https://github.com/user-attachments/assets/9a3edd0f-175a-4aca-97aa-e79ad24855d4" />

## Performance/efficiency considerations
Results - Total inference time on 2280 images: 12.2345 seconds
          Average inference time per image: 0.005366 seconds

- Used ResNet-50 model for higher accuracy, accepting longer training time and more compute.
- Enabled mixed precision training (AMP) to speed up training and reduce memory on CUDA GPUs.
- Optimized batch size and epochs to balance accuracy and training duration.
- Used PyTorch DataLoader with multiple workers and pin_memory=True for faster data loading.
- Saved best model checkpoints per epoch to avoid retraining from scratch.
- Automatic device selection (GPU if available, else CPU).
- Recommended cloud GPU (e.g., Google Colab with T4) to speed up training significantly compared to CPU.
- Handled missing GPU gracefully with warnings and fallback to CPU.

## Project structure (root)
- split_dataset.py       : Script to split a folder with class subfolders into train/val/test
- train.py               : Train script using PyTorch + pretrained ResNet-50
- test_model.py          : Evaluate the saved model on data/test and save confusion matrix
- utils.py               : Utilities (PDF/URL extraction, model loading, inference)
- app.py                 : Streamlit application to extract and classify images
- requirements.txt       : Python dependencies
- models/                : Saved model files (models/resnet50_trained.pth)
- data/                  : Empty by default. After running split_dataset.py it will contain train/val/test subfolders.
- output/                : Classification outputs (output/medical, output/non_medical)

## Expected input dataset structure (before splitting)
Provide a folder (example: `all_images`) with two subfolders named exactly:
- `medical/`      : Put medical images here. **Only** include images of these four modalities: X-ray, MRI, CT, Ultrasound.
- `non_medical/`  : Put non-medical images here (landscapes, animals, architecture, etc.)

Example:
```
all_images/
  medical/
    img001.png
    img002.jpg
  non_medical/
    img100.jpg
    img101.png
```

After running the split script (example):
```
python split_dataset.py --input all_images --out data
```
you will get:
```
data/
  train/
    medical/
    non_medical/
  val/
    medical/
    non_medical/
  test/
    medical/
    non_medical/
```


## How to use
1. Create virtual environment and install:
   ```
   python -m venv venv
   venv\Scripts\activate ( on Windows only check commands for other os)
   pip install -r requirements.txt
   ```
2. Place your images into `all_images/medical` and `all_images/non_medical`.
3. Split dataset:
   ```
   python split_dataset.py --input all_images --out data
   ```
4. Train model (example):
   ```
   python train.py --data_dir data --epochs 5 --batch_size 16 --lr 1e-4
   ```
   Model saved to `models/resnet50_trained.pth`.  ## add you model path to yout test and app .py file
5. Evaluate on test set:
   ```
   python test_model.py --model models/resnet50_trained.pth --data_dir data 
   ```
6. Run Streamlit app:
   ```
   streamlit run app.py
   ```
   - In the app you can upload a PDF or enter a URL with images.
   - The app will extract, classify and allow you to download results.



## Folder Complete Structure

<img width="850" height="526" alt="image" src="https://github.com/user-attachments/assets/58bd597a-5cf4-41dc-8ddd-ae5815703c2c" />



## Links:- 

   - [Link to all_images dataset](https://drive.google.com/drive/folders/10T-3wRP8-Ps4szhYR8zacDDZOib6UmF9?usp=sharing)
   - [Link to data dataset after split.py operation on all_images](https://drive.google.com/drive/folders/1FVnHA9e4V31fjiOC3lXNtcb8-QdHHb8_?usp=sharing)
   - [Link to resnet50_traned model](https://drive.google.com/drive/folders/1gOaF0GEBBtUw1clAk-k27h6ZJDXGgP9X?usp=sharing)
   








