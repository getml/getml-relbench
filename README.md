# GetML RelBench Noteboooks

### The Problem with Predictive Analytics on Relational Data

Relational data, often found in databases with multiple linked tables, poses a significant hurdle for traditional machine learning algorithms that typically expect a single, flat table. Relational learning overcomes this by employing algorithms to engineer features and learn directly from these interconnected data structures. This approach unlocks valuable insights hidden within the relationships between data points, leading to more accurate prediction models.

### getML Relational Learning: FastProp

At the heart of [getML-community](https://github.com/getml/getml-community) lies [FastProp](https://getml.com/latest/user_guide/concepts/feature_engineering/#feature-engineering-algorithms-fastprop), our open-source algorithm specifically designed for efficient feature engineering on relational data. FastProp seamlessly transforms complex relational data into a single table format, making it compatible with any machine learning algorithm. This automation not only saves you valuable time and effort but also has the potential to reveal hidden patterns crucial for accurate predictions.

* **Unmatched Speed:** It is engineered for speed, surpassing many existing methods in benchmarks. ([See the results](https://github.com/getml/getml-community?tab=readme-ov-file#benchmarks)).
* **Simplicity:** FastProp seamlessly integrates with the MLOps ecosystem, making it incredibly easy to incorporate into your workflow.
* **Enhanced Productivity:** By streamlining the tedious process of feature engineering, getML FastProp allows you to focus on the business-critical aspects of your project, and not thousands of lines of SQL.

## Introducing RelBench

[RelBench](https://relbench.stanford.edu/), a project from SNAP (Stanford University), provides a standardized set of benchmark datasets and tasks for evaluating relational learning algorithms. It aims to accelerate research and development in this field by offering a common ground for research. They created two baselines for comparison:  

* **Manual Feature Engineering by an expert:**  
In RelBench, [human data scientists](https://github.com/snap-stanford/relbench-user-study/) manually engineered features using their domain expertise and knowledge of relational databases. This involved carefully selecting, aggregating, and transforming data from multiple tables to create informative features.

* **Graph based neural networks:**  
Relational Deep Learning (RDL) represents a category of approaches within relational learning that is [developed at SNAP](https://github.com/snap-stanford/relbench/tree/main/examples), which leverages mostly graph neural networks to learn features from relational data.


### Let's Bring it Together

We've been busy putting [getML FastProp](https://github.com/getml/getml-community) to the test on RelBench, and the results are [quite impressive](#get-started-with-getml-fastprop)! We've already surpassed both RDL and human baselines on some tasks, and we believe there's plenty more potential to unlock.

### The Results

| **Dataset**                                                     | **Task**                                                                          | **PR's & Submissions**                                                                       | **Task + Measure**     | **Score getML**                  | **Score RDL** | **Score Human** |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------- | -------------------------------- | ------------- | --------------- |
| [rel-amazon](https://relbench.stanford.edu/datasets/rel-amazon) | [item-churn](https://relbench.stanford.edu/datasets/rel-amazon/#item-churn)       | [az-item-churn.ipynb](https://github.com/getml/getml-relbench/blob/main/az-item-churn.ipynb) | classification (AUROC) | [**0.831**](az-item-churn.ipynb) | 0.828         | 0.818           |
| [rel-hm](https://relbench.stanford.edu/datasets/rel-hm)         | [user-churn](https://relbench.stanford.edu/datasets/rel-hm/#user-churn)           | [hm-churn.ipynb](https://github.com/getml/getml-relbench/blob/main/hm-churn.ipynb)           | classification (AUROC) | [**0.707**](hm-churn.ipynb)      | 0.699         | 0.690           |
|                                                                 | [item-sales](https://relbench.stanford.edu/datasets/rel-hm/#item-sales)           | [hm-item.ipynb](https://github.com/getml/getml-relbench/blob/main/hm-item.ipynb)             | regression (MAE)       | [**0.031**](hm-item.ipynb)       | 0.056         | 0.036           |
