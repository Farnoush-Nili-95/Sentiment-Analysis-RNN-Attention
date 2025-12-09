Sentiment Analysis with RNNs vs. Attention

Author: Farnoush Nilizadeh

Course: Neural Computation

Year: 2025

Project Overview

This project investigates the impact of attention mechanisms on Recurrent Neural Networks (RNNs) for sentiment analysis. I implemented two models to classify IMDB movie reviews:

Baseline: A Bidirectional LSTM using the final hidden state.

Attention Model: A Bidirectional LSTM augmented with a custom attention layer.

Key Results

The attention mechanism significantly outperformed the baseline, proving that the ability to "focus" on specific words improves sentiment classification.

Baseline LSTM Accuracy: 74.97%

LSTM with Attention Accuracy: 88.66%


The Journey & Challenges (Adversity)

This project was not without challenges.

Library Issues: I initially attempted to use torchtext, but discovered it was deprecated and caused significant compatibility errors. I had to pivot and rewrite my entire data pipeline using the Hugging Face datasets library.

Tensor Dimensions: Implementing the attention mechanism from scratch (calculating the context vector using torch.bmm) resulted in several dimension mismatch errors. Debugging the shapes of the tensors at each step was a crucial learning experience.

Despite these hurdles, I successfully built a working attention model that provides interpretable results.

How to Run

Install dependencies: pip install -r requirements.txt

Run the training script: python sentiment_analysis.py
