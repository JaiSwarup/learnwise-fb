FROM python:3.11-slim-bullseye

WORKDIR /app

RUN pip install poetry

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY . .

EXPOSE $PORT

CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]