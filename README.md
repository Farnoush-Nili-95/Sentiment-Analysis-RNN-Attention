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

Video Walkthrough
https://temple.zoom.us/rec/play/HfCW0Rcq8AzwVVz4E6C3CSkxdIuPslPK8ZDloVHxxXpa5phUkKwIedWzR8b_gJr7xacpJEhr58jO5cUN.bddIijcY5-x5mKuH?eagerLoadZvaPages=sidemenu.billing.plan_management&accessLevel=meeting&canPlayFromShare=true&from=my_recording&continueMode=true&componentName=rec-play&originRequestUrl=https%3A%2F%2Ftemple.zoom.us%2Frec%2Fshare%2FP3C9fo7Y_hqHCBKT3Yl08XMTG98k7Dcmu3ET-PlBc1_CJzIlicvPrTaAL8d-OT0k.ZwCHFcrtsiONCIXz

The Journey & Challenges (Adversity)

This project was not without challenges.

Library Issues: I initially attempted to use torchtext, but discovered it was deprecated and caused significant compatibility errors. I had to pivot and rewrite my entire data pipeline using the Hugging Face datasets library.

Tensor Dimensions: Implementing the attention mechanism from scratch (calculating the context vector using torch.bmm) resulted in several dimension mismatch errors. Debugging the shapes of the tensors at each step was a crucial learning experience.

Despite these hurdles, I successfully built a working attention model that provides interpretable results.

How to Run

Install dependencies: pip install -r requirements.txt

Run the training script: python sentiment_analysis.py
