# BainInterviewChallenge
Code for the Bain Admission Interview, for position of Associate Machine Learning Engineer.

# Context
Challenge details specified [here](Challenge.md).

Essentially, we have a model available in [this notebook](Property-Friends-basic-model.ipynb), and must re-factor this model, such that it follows best practices, whilst also making it available through an API, running in a Docker container.

# Proposal
1. Restructure model into a class. This grants several benefits, such as abstraction, ease-of-maintenance, high modularity, and many more.

2. Once the entire model pipeline (construct, fit, test) has been refactored, create a deployment using [FastAPI](https://fastapi.tiangolo.com/).

3. Once the deployment is done, make a Dockerfile that sets up the environment for use.

# Implementation notes

### 1. The Refactoring

During the refactoring of the model, a few things came to mind. It is stated that the "Client is happy with the current results", and as such, the output of the model was kept identical to the notebook (as is mandatory in any such refactoring). However, I'd like to point out that the model's pipeline, as it stands, has a MASSIVE problem. Due to a small bug in the code, one of the features used as input for the regression is the `price` column. In other words, the model is using the price to predict the price. Even with this issue, there are other problems, the model is not as robust as it could be. No cross-validation is done, no other regression models were tested and no feature engineering was done (in this notebook). This seems to me a very big problem as well, that needs to be addressed, as the quality of the model's predictions that currently seem to please the client will likely cease to do so in the near future.

Now, with the model refactor complete (available with test script in [/model/model.py](/model/model.py)), we move on to a FastAPI architecture that can serve this model.


