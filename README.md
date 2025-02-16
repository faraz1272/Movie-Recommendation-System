# Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3.0-red)
![NumPy](https://img.shields.io/badge/NumPy-1.21.0-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.2-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11.1-lightblue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-ff69b4)
![CMF](https://img.shields.io/badge/CMF-3.0.0-purple)

This project implements a **Movie Recommendation System** using collaborative filtering techniques, including **item-item** and **user-user** approaches, as well as **matrix factorization**. The system recommends movies to users based on their preferences and similarities with other users or movies.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Approaches](#approaches)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Future Work](#future-work)
9. [License](#license)

---

## Project Overview

The goal of this project is to build a recommendation system that suggests movies to users based on their historical ratings and preferences. The system uses collaborative filtering techniques, including:
- **Item-Item Collaborative Filtering**: Recommends movies similar to the ones a user has liked.
- **User-User Collaborative Filtering**: Recommends movies liked by users with similar preferences.
- **Matrix Factorization**: Decomposes the user-movie interaction matrix to identify latent factors and make recommendations.

---

## Dataset

The dataset consists of three files:
1. **`ratings.dat`**: Contains user-movie ratings in the format `UserID::MovieID::Rating::Timestamp`.
2. **`users.dat`**: Contains user demographic information in the format `UserID::Gender::Age::Occupation::Zip-code`.
3. **`movies.dat`**: Contains movie information in the format `MovieID::Title::Genres`.

### Dataset Statistics
- **Users**: 6,040
- **Movies**: 3,952
- **Ratings**: 1,000,209
- **Genres**: 18 unique genres (e.g., Action, Comedy, Drama)

---

## Features

### Data Preprocessing
- Extracted release year from movie titles.
- Handled missing and incorrect data.
- Binned age groups and mapped occupations to meaningful labels.
- Created interaction matrices for collaborative filtering.

### Exploratory Data Analysis (EDA)
- Analyzed the distribution of ratings, movie releases, and user demographics.
- Visualized top genres by age group and most-rated movies.

### Recommendation Techniques
1. **Item-Item Collaborative Filtering**:
   - Used Pearson correlation and cosine similarity to find similar movies.
2. **User-User Collaborative Filtering**:
   - Used cosine similarity to find similar users.
3. **Matrix Factorization**:
   - Used the `CMF` library to decompose the user-movie interaction matrix and generate recommendations.

---

## Approaches

1. **Item-Item Collaborative Filtering**:
   - For a given movie, the system recommends movies with the highest correlation or cosine similarity.

2. **User-User Collaborative Filtering**:
   - For a given user, the system recommends movies liked by users with similar preferences.

3. **Matrix Factorization**:
   - Decomposed the user-movie interaction matrix into latent factors to predict user preferences.

4. **Nearest Neighbors**:
   - Used K-Nearest Neighbors (KNN) with cosine similarity to recommend movies.

---

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
