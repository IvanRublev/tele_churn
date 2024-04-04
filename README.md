# Customer Trends

You can play with the dashboard on https://tele-churn.fly.dev/

In this project, we will analyze customer behavior using :
* Exploratory Data Analysis
* Customer Segmentation using RFM Analysis
* Market Basket Analysis

## Output and execution

* Exploratory data analysis is cached after calculation at run time
* Market basket analysis breakdowns are prepared and persisted to disk during application deployment

## How to run for local development

1. Rename the .envrc-example to .envrc and specify values for environment variables
2. Load environment variables use `direnv allow` in the project's directory
3. Make sure that you have python and poetry installed with `asdf install`
4. Install project dependencies with `make deps`
5. Run tests with `make tests`, run server with `make server`
