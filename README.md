# BainInterviewChallenge
Code for the Bain Admission Interview, for position of Associate Machine Learning Engineer.

# Context
Challenge details specified [here](Challenge.md).

Essentially, we have a model available in [this notebook](Property-Friends-basic-model.ipynb), and must re-factor this model, such that it follows best practices, whilst also making it available through an API, running in a Docker container.

# Proposal
1. Restructure model into a class. This grants several benefits, such as abstraction, ease-of-maintenance, high modularity, and many more.

2. Once the entire model pipeline (construct, fit, test) has been refactored, create a deployment using [FastAPI](https://fastapi.tiangolo.com/).

3. Once the deployment is done, make a Dockerfile that sets up the environment for use.



