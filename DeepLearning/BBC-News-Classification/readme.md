# BBC News Classification: Matrix Factorization vs Supervised Learning

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=flat&logo=kaggle&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green.svg)

## Project Overview

This project explores **unsupervised learning** using matrix factorization techniques for text classification, specifically comparing it with traditional supervised learning approaches. We work with BBC news articles to classify them into 5 categories: business, entertainment, politics, sport, and tech.

**Kaggle Competition**: [BBC News Classification](https://kaggle.com/competitions/learn-ai-bbc)

### Key Objectives
1. **Matrix Factorization**: Apply Non-negative Matrix Factorization (NMF) as an unsupervised approach
2. **Feature Engineering**: Extract meaningful features from text using TF-IDF
3. **Performance Analysis**: Compare unsupervised vs supervised learning performance  
4. **Data Efficiency**: Analyze performance with varying amounts of labeled data

## Academic Context

This project is part of a Deep Learning course assignment focusing on:
- **Unsupervised Learning**: Matrix factorization for topic discovery
- **Text Mining**: NLP techniques for document classification
- **Comparative Analysis**: Understanding when to use unsupervised vs supervised methods
- **Data Efficiency**: Performance analysis across different training data sizes

### Assignment Requirements
- **Part 1**: Matrix factorization implementation and comparison (80 points)
- **Part 2**: Analysis of sklearn's NMF limitations (20 points)
- **Deliverables**: Jupyter notebook, GitHub repository, Kaggle submission

## Dataset

| Attribute | Details |
|-----------|---------|
| **Training Set** | 1,490 labeled articles |
| **Test Set** | 735 unlabeled articles |
| **Categories** | business, entertainment, politics, sport, tech |
| **Evaluation** | Accuracy |
| **Text Length** | Variable (avg ~500 words) |


## Methodology

### 1. Text Preprocessing Pipeline
- **Cleaning**: Remove punctuation, normalize case, filter short words
- **TF-IDF Vectorization**: Extract numerical features with n-grams
- **Feature Selection**: Optimize vocabulary size and term frequency thresholds

### 2. Matrix Factorization (NMF)
- **Topic Discovery**: Identify latent topics in document corpus
- **Unsupervised Learning**: No labels used during topic extraction
- **Topic-Category Mapping**: Align discovered topics with actual categories
- **Hyperparameter Tuning**: Optimize number of components and regularization

### 3. Supervised Learning Baselines
- **Naive Bayes**: Probabilistic classifier optimized for text
- **Support Vector Machine**: Linear classifier with TF-IDF features
- **Random Forest**: Ensemble method for comparison

### 4. Comparative Analysis
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Data Efficiency**: Performance vs training data size
- **Overfitting Analysis**: Training vs cross-validation performance

## Key Findings

### Performance Comparison
| Approach | Training Accuracy | Cross-Validation | Data Requirement |
|----------|------------------|------------------|------------------|
| **NMF (Unsupervised)** | ~75-80% | N/A | No labeled data |
| **Naive Bayes** | ~95-98% | ~92-95% | Full labeled dataset |
| **SVM** | ~98-99% | ~94-96% | Full labeled dataset |
| **Random Forest** | ~99%+ | ~90-93% | Full labeled dataset |

### Data Efficiency Insights
- **Small Data (10-30%)**: Gap between supervised and unsupervised narrows
- **Large Data (70-100%)**: Supervised learning shows clear advantage
- **Crossover Point**: Supervised becomes superior with >50% labeled data


## Results Analysis

### Matrix Factorization Success
- **Topic Discovery**: Successfully identified 5 meaningful topics
- **Category Alignment**: Topics showed 60-80% alignment with actual categories
- **Interpretability**: Discovered topics provided insights into document themes
- **Scalability**: Approach scales well to large document collections

### Supervised Learning Performance
- **Naive Bayes**: Best overall performance for text classification
- **SVM**: Strong performance with linear kernel
- **Random Forest**: Good performance but prone to overfitting

### Data Efficiency Analysis
- **Learning Curves**: Supervised models benefit significantly from more data
- **Minimum Data Requirements**: ~30% labeled data needed for supervised advantage
- **Practical Implications**: Choose approach based on available labeled data

## Future Extensions

### Technical Improvements
1. **Advanced Topic Models**: LDA, BERTopic, or neural topic models
2. **Hybrid Approaches**: Semi-supervised learning combining both methods
3. **Deep Learning**: Transformer-based classification (BERT, RoBERTa)
4. **Ensemble Methods**: Combine multiple unsupervised and supervised models

### Domain Applications
1. **Multi-language**: Extend to non-English news classification
2. **Real-time**: Streaming classification for live news feeds
3. **Hierarchical**: Multi-level category classification
4. **Sentiment**: Combine topic and sentiment analysis

## References

### Academic Sources
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. *Journal of Machine Learning Research*, 3, 993-1022.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791.

### Technical Resources
- [Scikit-learn NMF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [TF-IDF Vectorization Guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Topic Modeling with NMF](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html)

### Competition Details
- **Source**: Bijoy Bose. BBC News Classification. https://kaggle.com/competitions/learn-ai-bbc, 2019. Kaggle.
- **Evaluation**: Classification accuracy on test set
- **Baseline**: Random classifier (~20% accuracy)

## Contact

- **Author**: Herman Justino
- **Program**: MSc Data Science
- **Course**: Deep Learning - Unsupervised Learning
- **Institution**: [University Name]

---

*This project demonstrates the practical application of matrix factorization techniques for text classification and provides insights into the trade-offs between unsupervised and supervised learning approaches in natural language processing.*
