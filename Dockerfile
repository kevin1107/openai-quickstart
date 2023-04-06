FROM python:3.11.2

ENV FLASK_APP manage.py
ENV FLASK_CONFIG docker

RUN adduser --disabled-password --gecos "" openai
USER openai

WORKDIR /home/openai

COPY requirements requirements
RUN python -m venv venv
RUN venv/bin/pip install -r requirements/docker.txt

COPY app app
COPY .env.example .env
COPY manage.py config.py boot.sh ./

# run-time configuration
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]