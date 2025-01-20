# Data Science Challenge - getML Community

Welcome to our [getML](https://getml.com/) Data Science Challenge! This challenge invites you to tackle a series of relational learning benchmark problems using [getML FastProp](https://getml.com/latest/user_guide/concepts/feature_engineering/#feature-engineering-algorithms-fastprop) — the [fastest open-source tool](https://github.com/getml/getml-community?tab=readme-ov-file#benchmarks) for automated feature engineering on relational and time-series data.

:dart: [How to participate](#this-is-how-you-participate)
<br>
:clipboard: [Available Challenges](#pick-a-challenge)
<br>
:rocket: [Get Started with getML FastProp](#get-started-with-getml-fastprop)

Ready to elevate your data science skills and make a real impact? Here's what awaits you:

* **Build better models faster:**  
Learn how to use getML FastProp (short for Fast Propositionalization) for automated feature extraction from relational and time-series data to deliver more accurate prediction models without manual SQL or deep business domain expertise. [getML-community](https://github.com/getml/getml-community) is open source (ELv2), so you can keep using it to turbocharge your next predictive analytics project and beyond!

* **Gain recognition and rewards:**  
Showcase your skills by surpassing benchmark scores with the help of getML, and receive €100 as a token of our gratitude for every accepted submission. Your contributions may also be featured in community spotlights and future publications!

* **Expand your network:**  
Connect with the getML dev team and other data scientists on our [Discord](https://discord.gg/B4cC9uZHdx). Share your knowledge, learn from others, and help shape the future of open-source automated feature engineering.

**Enjoying what you see?** 🌟 We'd love for you to star our main repo, [getML-community repository](https://github.com/getml/getml-community). It's a great way to stay updated and show your support for our work! 🙌

> [!NOTE]
> This challenge will be open for submissions starting on the **17th of January 2025 at 16:00 CET!**
> 
> We will periodically add new challenges until we run out of tasks. New challenges will be added on the **20th of January at 16:00 CET.**

## Our Motivation

At getML, we've seen firsthand how FastProp simplifies and accelerates feature engineering for relational and time-series data. By launching this challenge, we aim to share these benefits with the broader data science and engineering community.  

Our goal is twofold:  
1. **Empower you** to tackle relational data problems with ease by adding FastProp to your toolkit, enabling you to focus on insights rather than manual feature extraction.  
2. **Collaborate with the community** to gather feedback, learn from real-world use cases, and continue enhancing FastProp based on your experiences.

By participating, you'll not only advance your skills but also contribute to the development of cutting-edge, open-source tools that are shaping the future of learning on relational data.


## What is Relational Learning?

### The Problem with Relational Data

Relational data, often found in databases with multiple linked tables, poses a significant hurdle for traditional machine learning algorithms that typically expect a single, flat table. Relational learning overcomes this by employing algorithms to engineer features and learn directly from these interconnected data structures. This approach unlocks valuable insights hidden within the relationships between data points, leading to more accurate prediction models.

### getML FastProp

At the heart of [getML-community](https://github.com/getml/getml-community) lies [FastProp](https://getml.com/latest/user_guide/concepts/feature_engineering/#feature-engineering-algorithms-fastprop), our open-source algorithm specifically designed for efficient feature engineering on relational data. FastProp seamlessly transforms complex relational data into a single table format, making it compatible with any machine learning algorithm. This automation not only saves you valuable time and effort but also has the potential to reveal hidden patterns crucial for accurate predictions.

* **Unmatched Speed:** It is engineered for speed, surpassing many existing methods in benchmarks. ([See the results](https://github.com/getml/getml-community?tab=readme-ov-file#benchmarks)).
* **Simplicity:** FastProp seamlessly integrates with the MLOps ecosystem, making it incredibly easy to incorporate into your workflow.
* **Enhanced Productivity:** By streamlining the tedious process of feature engineering, getML FastProp allows you to focus on the business-critical aspects of your project, and not thousands of lines of SQL.

### Introducing RelBench

[RelBench](https://relbench.stanford.edu/), a project from SNAP (Stanford University), provides a standardized set of benchmark datasets and tasks for evaluating relational learning algorithms. It aims to accelerate research and development in this field by offering a common ground for research. They created two baselines for comparison:  

* **Manual Feature Engineering by an expert:**  
In RelBench, [human data scientists](https://github.com/snap-stanford/relbench-user-study/) manually engineered features using their domain expertise and knowledge of relational databases. This involved carefully selecting, aggregating, and transforming data from multiple tables to create informative features.

* **Graph based neural networks:**  
Relational Deep Learning (RDL) represents a category of approaches within relational learning that is [developed at SNAP](https://github.com/snap-stanford/relbench/tree/main/examples), which leverages mostly graph neural networks to learn features from relational data.


## Let's Bring it Together

### Take on the Challenge

We've been busy putting [getML FastProp](https://github.com/getml/getml-community) to the test on RelBench, and the results are [quite impressive](#get-started-with-getml-fastprop)! We've already surpassed both RDL and human baselines on two tasks, and we believe there's plenty more potential to unlock. Now, it's your turn to explore the power of getML and see what you can achieve.

This challenge invites you to apply your data science skills and creativity to a series of unsolved RelBench tasks. We encourage you to experiment with getML FastProp, combine it with your favorite machine learning models, and see if you can surpass the existing baselines – and maybe even our own scores!

- **This is what we are looking for:**
  * Effective use of getML: Demonstrate a good understanding of getML data models and how to tune them for optimal performance.
  * Performance: Aim to outperform at least one of the existing [RelBench baselines](#pick-a-challenge).
  * Sound and reproducible code: Well-structured, modern, and commented code that others can easily understand and execute.

- **What you should bring:**
  * Basic Python skills.
  * Experience with scikit-learn or similar data science libraries.
  * Time to get to know getML FastProp by using the [example notebooks](#get-started-with-getml-fastprop) and [user guide](https://getml.com/latest/user_guide/).

Remember, everything is allowed: use all tools available, search engines, LLMs, or ask us on [Discord](https://discord.gg/B4cC9uZHdx).

Ready to take on the challenge? Choose a task and start building your solution!


### This is How You Participate

1. **Pick an Unsolved Task:**  
   * Refer to the [table in this repo's README](#pick-a-challenge) or [issues](https://github.com/getml/getml-relbench/issues/7) to find tasks without an open Draft Pull Request (PR).  
   * You may have only one active pull request (PR) at a time and can complete a maximum of two tasks in total.

2. **Open a Draft PR:**  
   * Review the [Terms of Service & Licensing](#terms-of-service--licensing) section and confirm your acceptance by including the specified details in your submission.
   * Create a [Draft Pull Request](https://github.com/getml/getml-relbench/compare) on the [getML relbench](https://github.com/getml/getml-relbench) GitHub repository.  
   * Title your branch as well as the PR: `[dataset-name]-[taskname]` (e.g., `rel-amazon-user-churn`).  
   * This reserves the task for you. No one else can claim it while you are actively working on your open Draft PR.

3. **Build a Predictor that Uses FastProp:**  
   * Develop your solution using **[getML FastProp](https://getml.com/latest/reference/feature_learning/fastprop/)** for feature engineering.  
   * Document your code, pipeline, and reasoning clearly in the notebook.  
   * We encourage you to use a regression or classification model of your choice on top of the getML-generated features. (see our [examples](#get-started-with-getml-fastprop))
   * Any used libraries must be published under any [OSI approved license](https://opensource.org/licenses) or the [Elastic License](https://www.elastic.co/licensing/elastic-license)

4. **Stay Active:**  
   * Continuously push updates to your PR to show progress.  
   * If there’s no commit, discussion, or meaningful interaction to advance the task within 5 days, we may close the Draft PR to make the task available to others.  
   * We aim to respond to questions, provide feedback, or suggest improvements promptly. Any delays on our side will not count towards inactivity.  
   * We reserve the right to determine if a PR is being actively worked on.  

5. **Aim to Beat Existing Scores:**  
   * Compare your final metric against the existing [RelBench baselines](#pick-a-challenge) (RDL and Human).  
   * Strive to outperform at least one existing [RelBench baseline](#pick-a-challenge) (RDL or Human). We believe this is achievable for many of the challenges!
   * If you surpass both, that’s excellent. We will consider to merge your PR immediately.

6. **Get Your PR Merged and Address Feedback:**  
   * Submit a well-documented Jupyter Notebook that allows for easy reproduction of your results.
   * Once you feel that your notebook is ready for merging, mark your PR as **Ready for Review**.  
   * We’ll review your PR and provide feedback where needed.  
   * Your PR will be merged into the main branch if it meets the criteria and successfully addresses our feedback.

7. **After Merging the PR:**  
   * We will contact you for the [reward](#rewards--recognition) using the email address associated with your commit. Alternatively, you can [securely provide us with a contact email address](#how-to-receive-the-reward) to facilitate communication.  
   * If you need assistance, feel free to reach out to us on [Discord](https://discord.gg/B4cC9uZHdx) or send an [email](mailto:support@getml.com) to our support team.

---

### Rewards & Recognition

1. Earn a €100 gift for each successfully merged notebook that outperforms at least one baseline.  
2. With your consent, we will gladly associate your name and social media profiles with the results on our communication channels.

---

### How to Receive the Reward

We offer two methods to deliver the €100 gift: through [GitHub Sponsors](https://docs.github.com/en/sponsors/receiving-sponsorships-through-github-sponsors/about-github-sponsors-for-open-source-contributors) or via [PayPal](https://paypal.com).  
* We will contact you using the email address associated with your commit to gather the necessary details.  
* If you are using [GitHub’s privacy feature](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address) to keep your email private, or if you prefer us to contact you at a different email address, please follow these steps:  
   1. To ensure the reward is delivered to the correct person without requiring you to publicly share your email address, use our [encryption tool](https://getml.com/relbench/encrypt) to securely encrypt your email.
   2. Include the encrypted email address in your PR. This method allows us to maintain your privacy while verifying the recipient.


## Get Started with getML FastProp

We've prepared two example notebooks to help you get started:

* [`hm-churn.ipynb`](https://github.com/getml/getml-relbench/blob/main/hm-churn.ipynb): A classification example using the H&M dataset, showcasing how to predict customer churn
* [`hm-item.ipynb`](https://github.com/getml/getml-relbench/blob/main/hm-item.ipynb): A regression example using the H&M dataset, demonstrating how to forecast item sales using FastProp together with LightGBM and Optuna

These notebooks provide a practical introduction to the getML workflow, from data loading and preprocessing to pipeline construction and evaluation.

Furhter information can be found in the **[User Guide](https://getml.com/latest/user_guide/)** and the **[API Reference](https://getml.com/latest/reference/)**.


## Pick a Challenge

> [!NOTE]
> New challenges will be added on the **27th of January at 16:00 CET.**

| **Dataset**                                                     | **Task**                                                                             | **PR's & Submissions**    | **Task + Measure**     | **Score getML**             | **Score RDL** | **Score Human** |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --- | ---------------------- | --------------------------- | ------------- | --------------- |
| [rel-amazon](https://relbench.stanford.edu/datasets/rel-amazon) | [user-churn](https://relbench.stanford.edu/datasets/rel-amazon/#user-churn)          |     | classification (AUROC) | –                           | **0.704**     | 0.676           |
|                                                                 | [item-churn](https://relbench.stanford.edu/datasets/rel-amazon/#item-churn)          | [PR #12](https://github.com/getml/getml-relbench/pull/12) | classification (AUROC) | –                           | **0.828**     | 0.818           |
|                                                                 | [user-ltv](https://relbench.stanford.edu/datasets/rel-amazon/#user-ltv)              |     | regression (MAE)       | –                           | 14.313        | **13.928**      |
|                                                                 | [item-ltv](https://relbench.stanford.edu/datasets/rel-amazon/#item-ltv)              |     | regression (MAE)       | –                           | 50.053        | **41.122**      |
| [rel-f1](https://relbench.stanford.edu/datasets/rel-f1)         | [driver-dnf](https://relbench.stanford.edu/datasets/rel-f1/#driver-dnf)              |     | classification (AUROC) | –                           | **0.726**     | 0.698           |
|                                                                 | [driver-top3](https://relbench.stanford.edu/datasets/rel-f1/#driver-top3)            |     | classification (AUROC) | –                           | 0.755     | **0.824**           |
|                                                                 | [driver-position](https://relbench.stanford.edu/datasets/rel-f1/#driver-position)    |     | regression (MAE)       | –                           | 4.022         | **3.963**       |
| [rel-hm](https://relbench.stanford.edu/datasets/rel-hm)         | [user-churn](https://relbench.stanford.edu/datasets/rel-hm/#user-churn)              |  [hm-churn.ipynb](https://github.com/getml/getml-relbench/blob/main/hm-churn.ipynb) | classification (AUROC) | [**0.707**](hm-churn.ipynb) | 0.699         | 0.690           |
|                                                                 | [item-sales](https://relbench.stanford.edu/datasets/rel-hm/#item-sales)              |  [hm-item.ipynb](https://github.com/getml/getml-relbench/blob/main/hm-item.ipynb) | regression (MAE)       | [**0.031**](hm-item.ipynb)  | 0.056         | 0.036           |
| [rel-trial](https://relbench.stanford.edu/datasets/rel-trial)   | [study-outcome](https://relbench.stanford.edu/datasets/rel-trial/#study-outcome)     |     | classification (AUROC) | –                           | 0.686     | **0.720**           |
|                                                                 | [study-adverse](https://relbench.stanford.edu/datasets/rel-trial/#study-adverse)     |     | regression (MAE)       | –                           | 44.473        | **40.581**      |
|                                                                 | [site-success](https://relbench.stanford.edu/datasets/rel-trial/#site-success)       |     | regression (MAE)       | –                           | **0.400**     | 0.407           |


## Let's Get Started!

We're excited to see your innovative solutions and contributions to the getML community. Good luck, happy coding, and let's push the boundaries of automated feature engineering together!

**You can count on our support:** Join our [Discord](https://discord.gg/B4cC9uZHdx) for technical guidance, feedback on your approaches, and to interact with the getML dev team. We're committed to helping you succeed in this challenge.

### How to start and setup the environment

To ease participation in the challenge, we have already prepared a base environment for you. To start hacking, you just need to clone this [repository](https://github.com/getml/getml-relbench.git) or clone your forked version of the repository.

#### Linux
To get started on linux, just [install uv](https://docs.astral.sh/uv/getting-started/installation/) and leverage our [curated environment](pyproject.toml):
```sh
uv run jupyter lab
```
#### macOS and Windows
To get started on macOS and Windows, you first need to [start the getML docker service](https://getml.com/latest/install/packages/docker/):
```sh
curl https://raw.githubusercontent.com/getml/getml-community/1.5.0/runtime/docker-compose.yml | docker-compose up -f -
```
Afterwards, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and use the [provided environment](pyproject.toml) as above:
```sh
uv run jupyter lab
```

You are always free to use any other project manger of your choice but we recommend to stick to [uv](https://pypi.org/project/uv/).

### Memory and External Compute Resources

For larger tasks and datasets, the size of the data might exceed the memory capacity of your local machine. One solution is to use the [memory mapping feature](https://getml.com/latest/reference/engine/engine/#getml.engine.launch), but keep in mind that this might increase compute time.

We recommend leveraging virtual machines from major cloud providers to handle such workloads. Many providers offer free trials or starter budgets that should cover your needs:  

- [Google Cloud Free Trial](https://cloud.google.com/free/docs/free-cloud-features#free-trial)  
- [Azure Free Trial](https://azure.microsoft.com/en-us/pricing/offers/ms-azr-0044p)

For most datasets we recommend that you do the final run on a machine with **128GB memory** and **32 CPU cores**.

## Terms of Service & Licensing

For complete details regarding the terms of service, please refer to the [terms of service (TOS)](./TOS.md). This document outlines all the guidelines, rules, and conditions associated with participation, including data protection, eligibility, and usage of resources.

The submitted notebook needs to contain the below license information and acknowledgement of the [terms of service (TOS)](./TOS.md) in a dedicated cell:

```markdown
**License:**

All code in this notebook is licensed under the [MIT License](https://mit-license.org/).

Text and Images License: All non-code content (text, documentation, and images) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Terms of serivce:**
I, [GitHub username], hereby acknowledge that I have read, understood, and agree to the [Terms of Service of the getML Community Data Science Challenge](https://github.com/getml/getml-relbench/blob/main/TOS.md).
```


---

### **Happy coding!**
