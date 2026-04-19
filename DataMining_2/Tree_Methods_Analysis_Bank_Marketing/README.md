Performed: 
    Spring 2026

Project: 
    A Tree-Based Approach to Modeling Client Subscription Likelihood for Term Deposits

Data: 
    Included in repo (source: https://archive-beta.ics.uci.edu/dataset/222/bank+marketing)

Synopsis:
    This study evaluates the predictive efficacy of Decision Tree, Bagging, and Random Forest models in identifying bank term deposit subscribers. Analyzing 45,211 client records, the project focuses on mitigating majority-class bias caused by a significant dataset imbalance (88% non-subscribers).<br>
	Key InsightsSuperior Model: Random Forest outperformed all candidates, achieving the highest F1-Score (0.5765) and Recall (52.08%).Addressing Imbalance: While accuracy remained high across all models ($\approx 91\%$), the baseline Decision Tree missed over half of potential subscribers. Ensemble methods proved essential for capturing these "minority" instances.<br>
	Primary Drivers: Call duration was the most significant predictor of conversion, followed by secondary factors such as age and account balance.<br>
	Strategic ConclusionThe transition from standalone trees to ensemble frameworks significantly improves targeting accuracy. For financial institutions, adopting Random Forest logic provides a more robust and stable tool for identifying high-propensity clients, ultimately optimizing telemarketing efficiency and resource allocation.