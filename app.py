from transformers import pipeline
import torch
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/mealtime-labels', methods=['POST'])
def get_mealtime_labels():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    review_json = request.get_json(force=True)
    review = review_json["review"]
    
    mealtime_labels = ["breakfast", "lunch", "dinner", "late-night"]

    res = classifier(review, candidate_labels=mealtime_labels, multi_label=True)

    mealtime_keywords = ""

    for label, score in zip(res["labels"], res["scores"]):
        print(f"{label}: {score:.2f}")
        if score > 0.30:
            mealtime_keywords += label
            mealtime_keywords += ", "
    
    return jsonify(mealtime_keywords)

@app.route('/semantic-labels', methods=['POST'])
def get_semantic_labels():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    review_json = request.get_json(force=True)
    review = review_json["review"]
    
    semantic_labels = ["romantic", "hangout", "business casual", "family", "celebration", "quick bite"]

    res = classifier(review, candidate_labels=semantic_labels, multi_label=True)

    semantic_keywords = ""

    for label, score in zip(res["labels"], res["scores"]):
        print(f"{label}: {score:.2f}")
        if score > 0.30:
            semantic_keywords += label
            semantic_keywords += ", "
    
    return jsonify(semantic_keywords)

if __name__ == '__main__':
   app.run(port=5001)