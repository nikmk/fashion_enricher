import os
import json
import logging
from io import BytesIO

import requests
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import clip

from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

from utils.tags import style_tag_bank, text_prompts, tag_to_category, extract_tags, adaptive_clip_filter

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
INPUT_JSON_NAME = os.getenv("INPUT_JSON_NAME", "products.json")
CHECKPOINT_FILE = "checkpoint.txt"

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)
blob = bucket.blob(INPUT_JSON_NAME)
products_data = json.loads(blob.download_as_string())

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        processed = set(line.strip() for line in f)
else:
    processed = set()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
yolo_model = YOLO("yolov8n.pt")
clip_model, clip_preprocess = clip.load("ViT-B/32")
clip_model.eval()

enriched_batch = []
batch_size = 25
batch_count = 0

def upload_batch(batch, count):
    batch_blob_name = f"products_enriched_batch_{count}.json"
    for attempt in range(3):
        try:
            bucket.blob(batch_blob_name).upload_from_string(json.dumps(batch, indent=2), content_type="application/json")
            logging.info(f"✅ Uploaded batch {count} to GCS as {batch_blob_name}")
            return True
        except GoogleAPIError as e:
            logging.warning(f"Retry {attempt + 1}/3 failed to upload batch {count}: {e}")
    logging.error(f"❌ Failed to upload batch {count} after 3 retries")
    return False

for product in tqdm(products_data, desc="Processing products"):
    image_url = product.get("main_image")
    if not product.get("product_id"):
        import uuid
        product["product_id"] = str(uuid.uuid4())
        logging.info(f"Generated product_id for {product.get('product_name')}: {product['product_id']}")
    if not image_url:
        logging.info(f"Skipping product {product.get('product_name')} - No image URL.")
        continue
    if product.get("yolo_tags"):
        logging.info(f"Skipping product {product.get('product_name')} - YOLO tags already exist.")
        continue

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.thumbnail((512, 512))
    except Exception as e:
        logging.warning(f"Skipping product {product.get('product_name')} due to image fetch error: {e}")
        continue

    inputs = processor(image, text=None, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    caption_tags = extract_tags(caption)

    image_np = np.array(image)
    temp_path = "temp_image.jpg"
    Image.fromarray(image_np).save(temp_path)
    yolo_results = yolo_model(temp_path)
    yolo_tags = list(set([yolo_results[0].names[c] for c in yolo_results[0].boxes.cls.int().tolist()]))
    os.remove(temp_path)

    clip_image = clip_preprocess(image).unsqueeze(0)
    text_inputs = clip.tokenize(text_prompts)
    with torch.no_grad():
        image_features = clip_model.encode_image(clip_image)
        text_features = clip_model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        cosine_scores = (image_features @ text_features.T)[0]
    matched_tags = [text_prompts[i] for i, score in enumerate(cosine_scores) if score > 0.25]
    filtered_tags, _ = adaptive_clip_filter(matched_tags, caption, yolo_tags, clip_model, clip.tokenize)

    img_cv = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_reshape = img_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(img_reshape)
    dominant_colors = [tuple(map(int, color)) for color in kmeans.cluster_centers_]
    product_id = product["product_id"]

    product.update({
        "product_id": product_id,
        "caption": caption,
        "caption_tags": caption_tags,
        "yolo_tags": yolo_tags,
        "matched_style_tags": filtered_tags,
        "dominant_colors_rgb": dominant_colors,
        "all_combined_tags": list(set(caption_tags + yolo_tags + filtered_tags))
    })
    enriched_batch.append(product)
    print(product["product_id"])

    with open(CHECKPOINT_FILE, "a") as ckpt:
        ckpt.write(product["product_id"] + "\n")

    if len(enriched_batch) >= batch_size:
        success = upload_batch(enriched_batch, batch_count)
        if success:
            enriched_batch = []
            batch_count += 1

# Upload any remaining products
if enriched_batch:
    upload_batch(enriched_batch, batch_count)
