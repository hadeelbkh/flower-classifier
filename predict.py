import argparse
import os
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

def process_image(image):
    image_size = (224, 224)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image.numpy()

# Predict the top K classes
def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.array(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(processed_image)
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]

    return top_k_probs, top_k_indices


def display_predictions(top_k_probs, top_k_classes, class_names):
    print('Flower labels with their probabilities:')
    for i in range(len(top_k_classes)):
        class_id = top_k_classes[i]
        flower_name = class_names.get(str(class_id), f"Class {class_id}") if class_names else f"Class {class_id}"
        print(f"{i+1}: {flower_name} - Probability: {top_k_probs[i]:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Use the trained model to predict the flowers' names")
    parser.add_argument('image_path', type=str, help='Provide the path to the image file')
    parser.add_argument('model_path', type=str, help='Provide the path to the trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Provide the number of top K classes')
    parser.add_argument('--category_names', type=str, help="Provide the path to JSON file that maps labels to flowers' names")

    args = parser.parse_args()

    # Load class names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = {}

    model = load_model(args.model_path)

    top_k_probs, top_k_classes = predict(args.image_path, model, args.top_k)

    display_predictions(top_k_probs, top_k_classes, class_names)

if __name__ == "__main__":
    main()
