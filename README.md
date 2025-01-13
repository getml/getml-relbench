# getML Community x RelBench Challenge

Welcome to the [getML](https://getml.com/) Community x RelBench Challenge! This challenge invites you to tackle a series of relational learning benchmark problems using [getML FastProp](https://getml.com/latest/user_guide/concepts/feature_engineering/#feature-engineering-algorithms-fastprop) — the [fastest open-source tool](https://github.com/getml/getml-community?tab=readme-ov-file#benchmarks) for automated feature engineering on relational and time-series data.

Ready to elevate your data science skills and make a real impact? Here's what awaits you:

* **Build better models faster:** Learn how to use getML FastProp (short for Fast Propositionalization) for automated features extraction from relational and time-series data to deliver more accurate prediction models without the likes of manual SQL or deep business domain expertise. [getml-community](https://github.com/getml/getml-community) is open source (ELv2), so you can keep using it to turbo-charge your next predictive analytics project and beyond!
* **Gain recognition and rewards:**  Showcase your skills, climb the leaderboards, and earn a €100 gift card for every accepted submission. Your contributions could be featured in community spotlights and future publications!
* **Expand your network:**  Connect with the getML dev team and other data scientists on our [Discord](https://discord.gg/B4cC9uZHdx). Share your knowledge, learn from others, and help shape the future of open-source automated feature engineering.


> This challenge will be open for submissions starting from the 20th of January 2025!

## What is Relational Learning?

### The Problem with Relational Data

Relational data, often found in databases with multiple linked tables, poses a significant hurdle for traditional machine learning algorithms that typically expect a single, flat table. Relational learning overcomes this by employing algorithms to engineer features and learn directly from these interconnected data structures. This approach unlocks valuable insights hidden within the relationships between data points, leading to more accurate prediction models.

### getML FastProp

At the heart of [getML-community](https://github.com/getml/getml-community) lies [FastProp](https://getml.com/latest/user_guide/concepts/feature_engineering/#feature-engineering-algorithms-fastprop), our open-source algorithm specifically designed for efficient feature engineering on relational data. FastProp, short for fast propositionalization, seamlessly transforms complex relational data into a single table format, making it compatible with any machine learning algorithm. This automation not only saves you valuable time and effort but also has the potential to reveal hidden patterns crucial for accurate predictions.

* **Unmatched Speed:**  It's engineered for speed, surpassing many existing methods in benchmarks. ([See the results](https://github.com/getml/getml-community?tab=readme-ov-file#benchmarks)).
* **Simplicity:** FastProp seamlessly integrates with the MLOps ecosystem, making it incredibly easy to incorporate into your workflow.
* **Enhanced Productivity:** By streamlining the tedious process of feature engineering, getML FastProp allows you to focus on the business critical aspects of your project, and not thousands of lines of SQL.


### Introducing RelBench

[RelBench](https://relbench.stanford.edu/), a project from SNAP (Stanford University), provides a standardized set of benchmark datasets and tasks for evaluating relational learning algorithms. It aims to accelerate research and development in this field by offering a common ground for comparison. 

* **How did human data scientists work?** In RelBench, [human data scientists](https://github.com/snap-stanford/relbench-user-study/) manually engineered features using their domain expertise and knowledge of relational databases. This involved carefully selecting, aggregating, and transforming data from multiple tables to create informative features.
* **What is RDL?** Relational Deep Learning represents a category of approaches within relational learning that leverages mostly graph neural networks to learn features from relational data.


## Let's Bring it Together - The getML x Relbench Challenge

### Take on the Challenge

We've been busy putting [getML FastProp](https://github.com/getml/getml-community) to the test on RelBench, and the results are [quite impressive](#get-started-with-getml-fastprop)!  We've already surpassed both RDL and human baselines on two tasks, and we believe there's plenty more potential to unlock. Now, it's your turn to explore the power of getML and see what you can achieve.

This challenge invites you to apply your data science skills and creativity to a series of unsolved RelBench tasks. We encourage you to experiment with getML FastProp, combine it with your favorite machine learning models, and see if you can surpass the existing baselines – and maybe even our own scores!

- **This is what we are looking for:**
    * Effective use of getML: Demonstrate a good understanding of getML data models and how to tune them for optimal performance.
    * Performance: Aim to outperform at least one of the existing [RelBench baselines](#pick-a-challenge).
    * Sound and reproducible code:  Well-structured, modern and commented code that others can easily understand and execute.

Ready to take on the challenge? Choose a task and start building your solution!

### This is How You Participate

1. **Pick an Unsolved Task:**
   * Refer to the [table in this repo's README](#pick-a-challenge) or [issues](https://github.com/getml/getml-relbench/issues/7) to find tasks without an open Draft Pull Request (PR).
2. **Open a Draft PR:** 
   * Create a [Draft Pull Request](https://github.com/getml/getml-relbench/compare) on the [getML relbench](https://github.com/getml/getml-relbench) GitHub repository.
   * Title it: `[dataset-name]-[taskname]` (e.g., `rel-amazon-user-churn`).
   * This reserves the task for you. No one else can claim it while your Draft PR is open.
3. **Build a Predictor that Uses FastProp:**
   * Develop your solution, using **[getML FastProp](https://getml.com/latest/reference/feature_learning/fastprop/)** for feature engineering.
   * Document your code, pipeline, and reasoning clearly in the notebook.
   * We highly encourage you to use a regression or classification model of your choice on top of the generated features.
4. **Stay Active:**
   * Push updates to your PR on a rolling basis.
   * If there's no activity for 5 days, we might close the Draft PR to free up the task for others.
   * Every meaningful commit, discussion or interaction resets the inactivty timer.
5. **Aim to Beat Existing Scores:**
   * Compare your final metric against the existing [RelBench baselines](#pick-a-challenge) (RDL and Human).
   * If you surpass at least one, that's excellent\!
   * If not, don't worry. We'll provide feedback and support.
6. **Address Feedback:**
   * We'll review your Draft PR and provide comments, suggestions, or questions.
   * We aim to review on a rolling basis and address your Draft PR within 7 days.
   * Once you've addressed the feedback, mark your PR as "Ready for Review".
7. **Get Your PR Merged:**
   * Your PR will be merged into the main branch if it meets the criteria and successfully addresses our feedback.
   * You will receive a €100 voucher as a thank you for your contribution.
   * You will receive public credit for your submission, and your name will be tied to it.
   * You will receive a notable mention in any follow-up studies or publications related to this challenge.


### Rewards & Recognition

1.  €100 Gift Card: A €100 gift card for each successfully merged notebook that beats at least one baseline.
2.  Community Spotlight: We'll announce your contribution on our communication channels, only if you agree.


## Get Started with getML FastProp

We've prepared two example notebooks to help you get started:

* [`hm-churn.ipynb`](https://github.com/getml/getml-relbench/blob/main/hm-churn.ipynb): A classification example using the H&M dataset, showcasing how to predict customer churn.
* [`hm-item.ipynb`](https://github.com/getml/getml-relbench/blob/2-benchmark-challenge/hm-item.ipynb): A regression example using the H&M dataset, demonstrating how to forecast item sales using FastProp together with LightGBM and Optuna.

These notebooks provide a practical introduction to the getML workflow, from data loading and preprocessing to pipeline construction and evaluation.

Furhter information can be found in the **[User Guide](https://getml.com/latest/user_guide/)** and the **[API Reference](https://getml.com/latest/reference/)**.


## Submission Criteria
 * **Participation**: Submit a maximum of two PRs.
 * **FastProp is Key**: Utilize [getML FastProp](https://getml.com/latest/reference/feature_learning/fastprop/#getml.feature_learning.FastProp) for automated feature engineering in your solutions.
 * **Performance Goal**: Strive to outperform at least one existing [RelBench baseline](#pick-a-challenge) (RDL or Human). We believe this is achievable for many of the challenges!
 * **Collaboration**: Actively participate by addressing feedback provided on your PR.
 * **Reproducibility**: Submit a well-documented Jupyter Notebook that allows for easy reproduction of your results.
 * **Leaderboard**: Acknowledge in your PR that your score and name (if you choose to share it) may appear in public leaderboard announcements.
 * **Licensing**: The submitted notebook needs to contain the below license infromation in a dedicated cell:


> Code License: All code in this notebook is licensed under the [MIT License](https://mit-license.org/).
Text and Images License: All non-code content (text, documentation, and images) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).



## Pick a Challenge

| **Dataset**                                                     | **Task**                                                                             | **PR's & Submissions**    | **Task + Measure**     | **Score getML**             | **Score RDL** | **Score Human** |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --- | ---------------------- | --------------------------- | ------------- | --------------- |
| [rel-amazon](https://relbench.stanford.edu/datasets/rel-amazon) | [user-churn](https://relbench.stanford.edu/datasets/rel-amazon/#user-churn)          |     | classification (AUROC) | –                           | **0.704**     | 0.676           |
|                                                                 | [item-churn](https://relbench.stanford.edu/datasets/rel-amazon/#item-churn)          |     | classification (AUROC) | –                           | **0.828**     | 0.818           |
|                                                                 | [user-ltv](https://relbench.stanford.edu/datasets/rel-amazon/#user-ltv)              |     | regression (MAE)       | –                           | 14.313        | **13.928**      |
|                                                                 | [item-ltv](https://relbench.stanford.edu/datasets/rel-amazon/#item-ltv)              |     | regression (MAE)       | –                           | 50.053        | **41.122**      |
| [rel-avito](https://relbench.stanford.edu/datasets/rel-avito)   | [ad-ctr](https://relbench.stanford.edu/datasets/rel-avito/#ad-ctr)                   |     | regression (MAE)       | –                           | **0.041**     | –               |
|                                                                 | [user-clicks](https://relbench.stanford.edu/datasets/rel-avito/#user-clicks)         |     | classification (AUROC) | –                           | **0.659**     | –               |
|                                                                 | [user-visits](https://relbench.stanford.edu/datasets/rel-avito/#user-visits)         |     | classification (AUROC) | –                           | **0.662**     | –               |
| [rel-event](https://relbench.stanford.edu/datasets/rel-event)   | [user-attendance](https://relbench.stanford.edu/datasets/rel-event/#user-attendance) |     | regression (MAE)       | –                           | **0.258**     | –               |
|                                                                 | [user-repeat](https://relbench.stanford.edu/datasets/rel-event/#user-repeat)         |     | classification (AUROC) | –                           | **0.769**     | –               |
|                                                                 | [user-ignore](https://relbench.stanford.edu/datasets/rel-event/#user-ignore)         |     | classification (AUROC) | –                           | **0.816**     | –               |
| [rel-f1](https://relbench.stanford.edu/datasets/rel-f1)         | [driver-dnf](https://relbench.stanford.edu/datasets/rel-f1/#driver-dnf)              |     | classification (AUROC) | –                           | **0.726**     | 0.698           |
|                                                                 | [driver-top3](https://relbench.stanford.edu/datasets/rel-f1/#driver-top3)            |     | classification (AUROC) | –                           | 0.755     | **0.824**           |
|                                                                 | [driver-position](https://relbench.stanford.edu/datasets/rel-f1/#driver-position)    |     | regression (MAE)       | –                           | 4.022         | **3.963**       |
| [rel-hm](https://relbench.stanford.edu/datasets/rel-hm)         | [user-churn](https://relbench.stanford.edu/datasets/rel-hm/#user-churn)              |  [hm-churn.ipynb](https://github.com/getml/getml-relbench/blob/main/hm-churn.ipynb) | classification (AUROC) | [**0.707**](hm-churn.ipynb) | 0.699         | 0.690           |
|                                                                 | [item-sales](https://relbench.stanford.edu/datasets/rel-hm/#item-sales)              |  [hm-item.ipynb](https://github.com/getml/getml-relbench/blob/main/hm-item.ipynb) | regression (MAE)       | [**0.031**](hm-item.ipynb)  | 0.056         | 0.036           |
| [rel-stack](https://relbench.stanford.edu/datasets/rel-stack)   | [user-engagement](https://relbench.stanford.edu/datasets/rel-stack/#user-engagement) |     | classification (AUROC) | –                           | **0.906**     | 0.903           |
|                                                                 | [user-badge](https://relbench.stanford.edu/datasets/rel-stack/#user-badge)           |     | classification (AUROC) | –                           | **0.889**     | 0.862           |
|                                                                 | [post-votes](https://relbench.stanford.edu/datasets/rel-stack/#post-votes)           |     | regression (MAE)       | –                           | 0.065         | 0.065           |
| [rel-trial](https://relbench.stanford.edu/datasets/rel-trial)   | [study-outcome](https://relbench.stanford.edu/datasets/rel-trial/#study-outcome)     |     | classification (AUROC) | –                           | 0.686     | **0.720**           |
|                                                                 | [study-adverse](https://relbench.stanford.edu/datasets/rel-trial/#study-adverse)     |     | regression (MAE)       | –                           | 44.473        | **40.581**      |
|                                                                 | [site-success](https://relbench.stanford.edu/datasets/rel-trial/#site-success)       |     | regression (MAE)       | –                           | **0.400**     | 0.407           |



## Let's Get Started!

We're excited to see your innovative solutions and contributions to the getML community. Good luck, happy coding, and let's push the boundaries of automated feature engineering together!

**You can count on our support:** Join our [Discord](https://discord.gg/B4cC9uZHdx) for technical guidance, feedback on your approaches, and to interact with the getML dev team. We're committed to helping you succeed in this challenge.