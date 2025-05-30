from transformers import pipeline
import torch
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/restaurant-labels', methods=['POST'])
def get_semantic_labels():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    review_json = request.get_json(force=True)
    review = review_json["text"]
    
    semantic_labels = ["comfort food", "quiet", "fast", "romantic", "gourmet", "budget", "casual", "relaxed", "dessert", "group", "loud", "lively"]

    res = classifier(review, candidate_labels=semantic_labels, multi_label=True)

    semantic_keywords = ""

    for label, score in zip(res["labels"], res["scores"]):
        print(f"{label}: {score:.2f}")
        if score > 0.75:
            semantic_keywords += label
            semantic_keywords += ", "
    
    return jsonify(semantic_keywords)

@app.route('/mood-labels', methods=['POST'])
def get_semenatic():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    review_json = request.get_json(force=True)
    review = review_json["text"]
    
    semantic_labels = ["stressed", "romantic", "busy", "celebration", "relaxed"]

    res = classifier(review, candidate_labels=semantic_labels, multi_label=True)

    max_index = res['scores'].index(max(res['scores']))

    # Get the top label
    top_mood = res['labels'][max_index]
    
    return jsonify(top_mood)

if __name__ == '__main__':
   app.run(port=5001)