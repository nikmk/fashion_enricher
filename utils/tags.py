

import re
import torch
import numpy as np

style_tag_bank = {
    "garment": [
        "a woman in a swimsuit",
        "a woman in a bikini",
        "a traditional Indian lehenga",
        "a kurta outfit",
        "a formal blazer",
        "a button-down shirt",
        "an elegant gown",
        "a western dress",
        "a saree",
        "a pair of trousers",
        "a pair of shorts",
        "a pair of jeans",
        "a pleated skirt",
        "a crop top outfit",
        "a sleeveless tank top",
        "a one-piece jumpsuit"
    ],
    "style": [
        "boho aesthetic",
        "barbiecore trend",
        "grunge fashion",
        "chic ensemble",
        "formal styling",
        "casual street look",
        "resortwear vibe",
        "power dressing",
        "business casual look",
        "quiet luxury outfit",
        "preppy school style",
        "urban streetwear fashion"
    ],
    "theme": [
        "floral pattern",
        "animal print design",
        "tie-dye art",
        "checkered graphics",
        "abstract shapes",
        "ethnic cultural motifs",
        "tropical beach theme",
        "futuristic cyber aesthetic",
        "retro disco style",
        "vintage nostalgia theme",
        "bold artistic prints"
    ],
    "material": [
        "pure cotton fabric",
        "linen blend material",
        "smooth polyester",
        "transparent mesh",
        "shiny satin",
        "faded denim cloth",
        "delicate lace",
        "soft chiffon",
        "rich silk",
        "plush velvet"
    ],
    "fit": [
        "structured silhouette",
        "relaxed oversized fit",
        "tailored precision cut",
        "body-hugging fitted look",
        "asymmetrical edge",
        "wrapped waistline"
    ],
    "color": [
        "jet black hue",
        "crisp white tone",
        "sunny yellow shade",
        "bright red palette",
        "deep blue tint",
        "lush green color",
        "pastel dreamy hues",
        "neon vibrant glow",
        "earthy natural tones"
    ],
    "detail": [
        "cut-out accents",
        "cinched belted waist",
        "multi-layered structure",
        "soft ruffled trims",
        "finely pleated folds",
        "wrap tie embellishment",
        "halter neckline",
        "strapless upper body"
    ]
}

text_prompts = sum(style_tag_bank.values(), [])
tag_categories = style_tag_bank

tag_to_category = {}
for category, tag_list in tag_categories.items():
    for tag in tag_list:
        tag_to_category[tag] = category

def extract_tags(caption):
    words = re.findall(r'\b\w+\b', caption.lower())
    stopwords = {"a", "the", "in", "on", "of", "with", "and", "at", "to", "an"}
    return list(set([word for word in words if word not in stopwords]))

def adaptive_clip_filter(matched_tags, caption, yolo_tags, clip_model, clip_tokenizer):
    if not matched_tags:
        return [], []

    context_string = caption + " " + " ".join(yolo_tags)
    context_tokens = clip_tokenizer([context_string], truncate=True)
    tag_tokens = clip_tokenizer(matched_tags, truncate=True)

    with torch.no_grad():
        context_feat = clip_model.encode_text(context_tokens)
        context_feat /= context_feat.norm(dim=-1, keepdim=True)

        tag_feats = clip_model.encode_text(tag_tokens)
        tag_feats /= tag_feats.norm(dim=-1, keepdim=True)

        sims = (tag_feats @ context_feat.T).squeeze(1).cpu().numpy()

    if sims.size == 0:
        return [], []

    percentile_cutoff = np.percentile(sims, 40)
    final_tags = []
    dropped_tags = []

    garment_scores = [(tag, sim) for tag, sim in zip(matched_tags, sims) if tag_to_category.get(tag) == "garment"]
    garment_scores.sort(key=lambda x: x[1], reverse=True)
    top_garment_tags = [tag for tag, _ in garment_scores[:2]]

    for tag, sim in zip(matched_tags, sims):
        category = tag_to_category.get(tag)

        if category in ["style", "theme", "material", "fit", "detail"]:
            final_tags.append(tag)
        elif category == "garment":
            if tag in caption or tag in yolo_tags or tag in top_garment_tags:
                final_tags.append(tag)
            else:
                dropped_tags.append(tag)
        else:
            if sim >= percentile_cutoff:
                final_tags.append(tag)
            else:
                dropped_tags.append(tag)

    return final_tags, dropped_tags