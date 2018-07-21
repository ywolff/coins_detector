FROM python:3.6.4

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install uwsgi

ADD requirements.txt /app
RUN pip install -r requirements.txt

ADD . /app

EXPOSE 1234

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:1234", "wsgi:app", "--reload", "--timeout", "120"]
