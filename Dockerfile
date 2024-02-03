# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN apt update -y; apt install -y \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    pkg-config \
    libgl1-mesa-glx \
    gcc-6 g++-6

WORKDIR /workspace
COPY . /workspace/
COPY ./requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

### fastapi
RUN pip install fastapi
RUN pip install "uvicorn[standard]"


# CMD tail -f /dev/null
