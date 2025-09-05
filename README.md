# Lyric Analysis Software

## Description
This project analyzes song lyrics to determine similarity based on both **TF-IDF text relevance** and **sentiment analysis**. Users can search for songs with similar emotional tone and word usage.

## Features
- Preprocesses lyrics: tokenization, stop word removal, stemming, and text normalization.  
- Computes TF-IDF scores for lyrics to evaluate textual similarity.  
- Normalizes sentiment scores using TextBlob to compare emotional content.  
- Combines sentiment and TF-IDF to return the top matching songs for a query.

## Tech Stack
- Python 3  
- Libraries: `pandas`, `NumPy`, `nltk`, `TextBlob`  
