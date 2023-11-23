FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . . 
RUN chmod +x .

CMD ["gunicorn", "-b", "0.0.0.0:6099", "app:app"]