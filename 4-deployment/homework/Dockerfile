FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# install pipenv
RUN pip install pipenv

# install dependencies
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system

# copy the app
COPY ["starter.py", "./"]

# run the app
# CMD ["python", "starter.py", "--year", "2023", "--month", "5"]
ENTRYPOINT ["python","starter.py"]