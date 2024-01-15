# BainInterviewChallenge
Code for the Bain Admission Interview, for position of Associate Machine Learning Engineer.

# Context
Challenge details specified [here](Challenge.md).

Essentially, we have a model available in [this notebook](Property-Friends-basic-model.ipynb), and must re-factor this model, such that it follows best practices, whilst also making it available through an API, running in a Docker container.

# How to use

If you wish to simply run the dockerized environment, then [install Docker](https://docs.docker.com/engine/install/), and then build and run the container.

To build:

    $ docker build -t re_chile_prediction_api .

Note that `re_chile_prediction_api` is the name for the container, and can be changed freely. Don't forget the dot at the end.

To run:

    $ docker run -p 8000:8000 re_chile_prediction_api

Also note that, without the -p clause, you will not be able to access the container from your machine. I am using port 8000 on my localhost, but if you wish to use another, change 8000:8000 to 8000:<YOUR-PORT>.

If you wish to run the project locally, you will need python (at least 3.8). Install dependencies listed on requirements.txt

    $ python3 -m pip install -r requirements.txt

Then, to train a new model, you should run

    $ python3 model/model.py

Once you have trained a new model, you can start the api using

    $ python3 api/main.py

Which will start the uvicorn server for FastAPI on port 8000.

### API Keys

To make requests to the api, you will need a new api key. For the sake of simplicity a placeholder route was created in the api to get you a fresh key. A get request to the `/new_key` route will return a fresh key, with a lifespan of 24h. Lifespan can be specified, as a float, in hours, via the query parameter keyLifespan, i.e. `/new_key/?keyLifespan=48` to generate a 48h key.

With a key in hands, one must make a post request, with the data that the model will need to make a prediction, whilst setting the `X-API-Key` header to the received api key, as done in the [example notebook](/api/api_test.ipynb).

The required formatting for the sent json can be found in the automatically generated documentation for the api, available in the `/docs` route once the api is running.

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

Once the basic logging was complete, I moved on to the API key solution. For the purposes of this project, I opted to make a simple key vault, in the `api/app/secrets` folder, included in the .gitignore. This key vault is very basic, and is implemented in the form of a csv. In the future this could easily be swapped with a SQL vault, or a different Docker container with a Key Vault solution. The implementation of this api key manager is done in the [key manager](/api/app/key_manager.py) file, and essentially only needs two methods, one to generate new keys (which come with a mandatory fixed lifespan) and one to validate a given key. The key manager also has its own separate log, in the same folder as the api, albeit with different filenames. The log structure is simpler, and is meant only to record the generation of new keys, and record every validation attempt and its result.

# 3. Dockerization

Dockerizing the application was very simple. A [Dockerfile](./Dockerfile) was created, which copies the content of this project into a container image. Then, when this image is run, it starts up the API service, and accepts requests via the usual default port 8000. A few tweaks to file references were made to fix some dependency issues.

# Final steps:

After finishing the implementation, I figured I'd take a shot at making a better model (at least a model in accordance with best practices).

The testing for this model can be found in the [model_development notebook](./model_development.ipynb), and in the end, I achieved a model with better accuracy, without relying on the actual target of the prediction.

The final model was a Random Forest Regressor, with some simple Feature Engineering. The final metrics achieved were:

|Model Name| RMSE | MAPE | MAE |
|-|-|-|-|
|Original|10254|40%|5859|
|My RFR|4420|12%|2136|