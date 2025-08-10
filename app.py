import os
import io
import zipfile
import tempfile
from typing import List, Dict
import requests
from urllib.parse import urljoin

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
import fitz 

from utils import load_checkpoint, ensure_dir  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Your_tarined_model_path"  # <- set your model path here

MODEL = load_checkpoint(MODEL_PATH, DEVICE)
st.success(f"Model loaded from {MODEL_PATH}")

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def classify_images(model: torch.nn.Module, images: List[Image.Image], batch_size: int = 16):
    transform = get_inference_transform()
    device = next(model.parameters()).device
    results = []
    tensors = []
    for img in images:
        tensors.append(transform(img).unsqueeze(0))
    if not tensors:
        return results
    dataset = torch.cat(tensors, dim=0)
    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            dataset = dataset.to(device)
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            top_probs, preds = torch.max(probs, dim=1)
            top_probs = top_probs.cpu().numpy()
            preds = preds.cpu().numpy()
            for p, c in zip(preds, top_probs):
                label = "medical" if int(p) == 0 else "non_medical"
                results.append((label, float(c)))
    return results

def extract_embedded_images_from_pdf(file_bytes: bytes, save_dir: str, min_size: int = 100) -> List[str]:
    ensure_dir(save_dir)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    saved = []
    for i in range(len(doc)):
        page = doc[i]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if pil_img.width < min_size or pil_img.height < min_size:
                continue  # skip tiny logos/icons
            out_path = os.path.join(save_dir, f"page{i+1}_img{img_index+1}.{image_ext}")
            pil_img.save(out_path)
            saved.append(out_path)
    doc.close()
    return saved

def download_images_from_url(page_url: str, save_dir: str, min_size: int = 100) -> List[str]:
    ensure_dir(save_dir)
    import bs4

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/116.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(page_url, headers=headers, timeout=15)
    r.raise_for_status()
    soup = bs4.BeautifulSoup(r.text, "html.parser")

    saved = []
    seen_urls = set()

    for idx, img in enumerate(soup.find_all("img")):
        src = img.get("src")
        if not src:
            src = img.get("data-src")
        if not src:
            continue

        if img.get("srcset"):
            srcset_items = img.get("srcset").split(",")
            largest = srcset_items[-1].strip().split(" ")[0]
            src = largest

        src = urljoin(page_url, src)

        if src in seen_urls:
            continue
        seen_urls.add(src)

        try:
            resp = requests.get(src, headers=headers, timeout=12)
            resp.raise_for_status()
            pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            if pil_img.width < min_size or pil_img.height < min_size:
                continue
            ext = "jpg" if "jpeg" in resp.headers.get("content-type", "").lower() else "png"
            out_path = os.path.join(save_dir, f"url_img_{len(saved)+1}.{ext}")
            pil_img.save(out_path)
            saved.append(out_path)
        except Exception:
            continue

    return saved


def make_csv_bytes(prediction_records: List[Dict]):
    import csv
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["filename", "label", "confidence"])
    for rec in prediction_records:
        writer.writerow([rec["filename"], rec["label"], f"{rec['confidence']:.6f}"])
    return out.getvalue().encode("utf-8")

def make_zip_bytes(filepaths: List[str], arcname_prefix=""):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in filepaths:
            arcname = os.path.join(arcname_prefix, os.path.basename(fp))
            z.write(fp, arcname)
    bio.seek(0)
    return bio.read()

st.title("Medical vs Non-Medical Image Classifier (PDF & URL)")

st.markdown("**Upload a PDF** (extracts embedded images only) or enter a webpage URL to scrape images.")

col1, col2 = st.columns(2)
with col1:
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
with col2:
    page_url = st.text_input("Webpage URL")

if st.button("Run classification"):
    with st.spinner("Processing..."):
        base_temp_dir = tempfile.mkdtemp(prefix="imgproc_")
        extracted_dir = os.path.join(base_temp_dir, "extracted")
        ensure_dir(extracted_dir)
        saved_files = []

        if uploaded_pdf:
            try:
                pdf_bytes = uploaded_pdf.read()
                pdf_saved = extract_embedded_images_from_pdf(pdf_bytes, extracted_dir)
                saved_files.extend(pdf_saved)
                st.success(f"Extracted {len(pdf_saved)} embedded images from PDF.")
            except Exception as e:
                st.error(f"PDF processing failed: {e}")

        if page_url:
            try:
                url_saved = download_images_from_url(page_url, extracted_dir)
                saved_files.extend(url_saved)
                st.success(f"Downloaded {len(url_saved)} images from URL.")
            except Exception as e:
                st.error(f"URL scraping failed: {e}")

        if len(saved_files) == 0:
            st.warning("No images found.")
        else:
            pil_images, filenames = [], []
            for fp in saved_files:
                try:
                    pil = Image.open(fp).convert("RGB")
                    pil_images.append(pil)
                    filenames.append(fp)
                except:
                    continue

            results = classify_images(MODEL, pil_images)
            records, med_files, non_med_files = [], [], []
            for fname, (label, conf) in zip(filenames, results):
                rec = {"filename": os.path.basename(fname), "filepath": fname,
                       "label": label, "confidence": conf}
                records.append(rec)
                if label == "medical":
                    med_files.append(fname)
                else:
                    non_med_files.append(fname)

            import pandas as pd
            st.session_state["predictions"] = records
            st.session_state["med_files"] = med_files
            st.session_state["non_med_files"] = non_med_files

            df = pd.DataFrame([{"filename": r["filename"], "label": r["label"], "confidence": r["confidence"]} for r in records])
            st.dataframe(df)

            cols = st.columns(4)
            for i, r in enumerate(records[:12]):
                with cols[i % 4]:
                    st.image(r["filepath"], caption=f"{r['filename']} | {r['label']} {r['confidence']:.2f}", use_column_width=True)

if "predictions" in st.session_state:
    csv_bytes = make_csv_bytes(st.session_state["predictions"])
    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    if st.session_state["med_files"]:
        med_zip = make_zip_bytes(st.session_state["med_files"], arcname_prefix="medical")
        st.download_button("Download MEDICAL images ZIP", data=med_zip, file_name="medical_images.zip", mime="application/zip")
    if st.session_state["non_med_files"]:
        non_zip = make_zip_bytes(st.session_state["non_med_files"], arcname_prefix="non_medical")
        st.download_button("Download NON-MEDICAL images ZIP", data=non_zip, file_name="non_medical_images.zip", mime="application/zip")
