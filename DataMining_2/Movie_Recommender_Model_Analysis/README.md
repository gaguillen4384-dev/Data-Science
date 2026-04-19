Performed: 
    Spring 2026

Project: 
    Evaluating Matrix Factorization Robustness in Highly Sparse Movie Recommendation Datasets

Data: 
    Included in repo:
	https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k

Synopsis:
    This project evaluates the efficacy of three matrix factorization architectures—Funk SVD, LFM-SGD, and Sparse Coding—in predicting user preferences within highly sparse datasets (93.7% sparsity). Utilizing the MovieLens 100K dataset, the study benchmarks these models across regression-based accuracy metrics (RMSE, MAE) and ranking-based retrieval metrics (NDCG, Precision, Recall). The results demonstrate a significant divergence between absolute rating precision and ranking proficiency. While all models reached a predictive floor in RMSE, Sparse Coding emerged as the superior framework, achieving a near-optimal NDCG of 0.99 and a Precision of 0.90. This highlights the model's robust ability to isolate salient features and generate high-quality "Top-N" recommendations despite noisy point-wise rating predictions.