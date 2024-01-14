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

### 2. The API

As an initial implementation, I opted to use FastAPI, given its robustness and ease of use. I created a basic application which has a single route, `/predict`, which allows only for POST requests, containing the necessary data (as of yet unencrypted). The sent data then is funneled into a saved version of the pre-trained model (this is done so that we need'nt train the model every time the server boots, and it is reasonable to assume that in production, we will use some more elaborate model, that will be saved and loaded in a more cloud-friendly way, so modularizing this seemed like the best option). The prediction is returned as a single float. In the future, a more elaborate response could be constructed, including error margins, uncertainties, or other such valuable information.

I then added some very simple logging using the native python module. This could be improved in a real deployment setting, using existing logging servers the client might already have, using encryption, or using rotating log files with backups, but for the purposes of this project, this seemed sufficient.

Logs are stored in the [logs folder](./logs/), and have a file_name with the day the server was started.

Log format is 
`PID:(process id) (TID:(thread id)) (log level name): [(human-readable timestamp)] (message)`

Once the basic logging was complete, I moved on to the API key solution. For the purposes of this project, I opted to make a simple key vault, in the `api/secrets` folder, included in the .gitignore. This key vault is very basic, and is implemented in the form of a csv. In the future this could easily be swapped with a SQL vault, or a different Docker container with a Key Vault solution. The implementation of this api key manager is done in the [key manager](/api/key_manager.py) file, and essentially only needs two methods, one to generate new keys (which come with a mandatory fixed lifespan) and one to validate a given key.