ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.9.8

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.2.0
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV SPARK_DRIVER_MEMORY=8G
ENV SPARK_EXECUTOR_CORES=12
ENV SPARK_EXECUTOR_MEMORY=8G
ENV SPARK_WORKER_CORES=12
ENV SPARK_WORKER_MEMORY=8G

COPY . .

#CMD [ "spark-submit", "./src/train.py"]
CMD ["python", "-m", "pytest", "tests", "--disable-pytest-warnings"]