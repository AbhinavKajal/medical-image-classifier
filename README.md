# medical-image-classifier
Image Classifier — A ResNet-50 based Streamlit app that classifies images as medical or non-medical. 

This repository contains a complete pipeline for building a binary image classifier (medical vs non-medical)
and a Streamlit app to extract images from PDFs / URLs and classify them.

## Project structure (root)
- split_dataset.py       : Script to split a folder with class subfolders into train/val/test
- train.py               : Train script using PyTorch + pretrained ResNet-50
- test_model.py          : Evaluate the saved model on data/test and save confusion matrix
- utils.py               : Utilities (PDF/URL extraction, model loading, inference)
- streamlit_app.py       : Streamlit application to extract and classify images
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
   Model saved to `models/resnet50_trained.pth`.
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










