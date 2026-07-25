import os
import random
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_classifier_model(model_path="CatsvsDogs.keras"):
    """
    Load and cache the trained Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found in working directory.")
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_input, target_size=(150, 150)):
    """
    Universal image preprocessing logic.
    Converts any image format (PNG, JPG, WEBP, BMP, GIF, RGBA, Grayscale, etc.)
    to RGB 150x150 tensor (1, 150, 150, 3).
    """
    if isinstance(image_input, (str, bytes, os.PathLike)):
        img = Image.open(image_input)
    elif hasattr(image_input, "read"): # Streamlit UploadedFile
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("Unsupported image input type.")
    
    # Handle EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Convert RGBA/Palette/Grayscale to RGB with white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        background = Image.new("RGB", alpha_img.size, (255, 255, 255))
        background.paste(alpha_img, mask=alpha_img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    
    original_img = img.copy()
    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(resized_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return original_img, resized_img, img_batch

def predict_pet(model, img_batch):
    """
    Runs model inference and returns class, raw score, and confidence percentage.
    0 = Cat, 1 = Dog
    """
    prediction = model.predict(img_batch, verbose=0)
    raw_score = float(prediction[0][0])
    
    if raw_score >= 0.5:
        predicted_class = "Dog"
        confidence = raw_score * 100.0
        cat_prob = (1.0 - raw_score) * 100.0
        dog_prob = raw_score * 100.0
    else:
        predicted_class = "Cat"
        confidence = (1.0 - raw_score) * 100.0
        cat_prob = (1.0 - raw_score) * 100.0
        dog_prob = raw_score * 100.0
        
    return predicted_class, confidence, raw_score, cat_prob, dog_prob

# Funny & Witty Reply Banks
FUNNY_CAT_QUOTES = [
    "Purr-fect! 100% Cat. Currently calculating the exact force required to knock your coffee off the desk. ☕💥",
    "Meow-velous! Definitely a Cat. Has exactly 0.00001% interest in whatever you are saying right now. 🐾",
    "Cat detected! 99% confidence. 1% chance it's a tiny majestic lion wearing an invisible crown. 👑",
    "Legally owns your house, your furniture, and your soul. You are merely the designated food opener. 🥫",
    "Cat Alert! Currently judging your life choices with 99.9% accuracy. 😼",
    "WARNING: Petting may trigger a sudden 180° trap bite after 3.5 seconds of purring! 😼⚡",
    "Definitely a Cat! Staring intently at a completely blank wall... seeing things we mere mortals cannot. 👻"
]

FUNNY_DOG_QUOTES = [
    "Woof! 100% Dog! Professional squirrel surveillance engineer & master of the 3 AM zoomies! 🐕💨",
    "Dog detected! Will trade 100% eternal loyalty for 1 tiny piece of bacon. 🥓",
    "Certified Good Boy/Girl! Warning: Hearing the word 'W-A-L-K' will trigger chaotic happiness! 🔔🔊",
    "Woof woof! Current mood: READY TO CHASE THE UNTAINTED TAIL OF DESTINY! 🌀",
    "Dog alert! Master of the 'I haven't been fed in 10 years' dramatic eyes while you eat pizza. 🍕",
    "100% Dog! Thinks the mail carrier is a villain plotting world domination. 📦🐾",
    "Paws up! Belly rubs are non-negotiable. Please apply gentle pressure immediately. 🐕❤️"
]

FUNNY_UNCERTAIN_QUOTES = [
    "Wait a sec... Is this a Cat, a Dog, or a fluffy potato wearing a tiny wig? 🥔",
    "My detective whiskers are twitching! Is this a secret agent pet disguised as household furniture? 🕵️‍♂️",
    "Hmm... 50% Cat, 50% Dog, 100% mysterious creature! Did a shapeshifter land in your living room? 🛸",
    "My AI brain is confused! Please hold a treat near the screen so I can smell it! 🥓🔍"
]

def get_anthropomorphic_mascot(predicted_class, confidence, threshold=50.0, seed=None):
    """
    Returns anthropomorphic mascot metadata, funny hilarious quotes, and SVG badges.
    """
    if seed is not None:
        random.seed(seed)

    is_uncertain = confidence < threshold
    
    if is_uncertain:
        quote = random.choice(FUNNY_UNCERTAIN_QUOTES)
        return {
            "name": "Professor Paw 🧐",
            "title": "The Puzzled Detective",
            "quote": quote,
            "mood": "uncertain",
            "badge_color": "#FF9800",
            "emoji": "🧐🔍",
            "funny_fact": f"Low Confidence Alert ({confidence:.1f}%): My radar is confused!"
        }
    elif predicted_class == "Cat":
        quote = random.choice(FUNNY_CAT_QUOTES)
        return {
            "name": "Detective Whiskers 🐱",
            "title": "Supreme Ruler & Cat Inspector",
            "quote": quote,
            "mood": "cat_mode",
            "badge_color": "#FF5252",
            "emoji": "😸🐾",
            "funny_fact": f"Feline Certainty: {confidence:.1f}% • Meow Factor: 10/10"
        }
    else: # Dog
        quote = random.choice(FUNNY_DOG_QUOTES)
        return {
            "name": "Inspector Barkley 🐶",
            "title": "Chief Good Boy & Dog Officer",
            "quote": quote,
            "mood": "dog_mode",
            "badge_color": "#0288D1",
            "emoji": "🐶🦴",
            "funny_fact": f"Canine Certainty: {confidence:.1f}% • Tail Wags: Unlimited"
        }
