.PHONY: deps lint shell server server_headless test docker_up docker_down

deps:
	brew install libomp
	poetry install

lint:
	poetry run ruff check . 

shell:
	poetry run python

server:
	poetry run streamlit run app.py

server_headless:
	poetry run streamlit run app.py --browser.serverAddress 0.0.0.0 --server.headless true

test:
	poetry run -- ptw -- -s -vv $(args)

test_once:
	poetry run pytest -s

docker_up:
	docker build -t tele-churn . && docker run -d -e STREAMLIT_SERVER_COOKIE_SECRET=$${STREAMLIT_SERVER_COOKIE_SECRET} -e STREAMLIT_SERVER_PORT=$${STREAMLIT_SERVER_PORT} -p $${STREAMLIT_SERVER_PORT}:$${STREAMLIT_SERVER_PORT} tele-churn

docker_down:
	docker ps -a -q --filter ancestor=tele-churn | xargs -I {} sh -c 'docker stop {} && docker rm {}'