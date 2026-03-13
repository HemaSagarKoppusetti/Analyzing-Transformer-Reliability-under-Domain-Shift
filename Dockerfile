FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install \
    transformers \
    datasets \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    tqdm \
    sentencepiece \
    accelerate

ENV PYTHONUNBUFFERED=1

CMD ["python"]