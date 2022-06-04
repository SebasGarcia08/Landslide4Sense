FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

# Set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install git
# Install CMake
#RUN apt install -y libprotobuf-dev protobuf-compiler
#RUN apt-get update && apt-get -y install cmake
RUN pip3 install pip==22.0.4

WORKDIR /app

COPY . .
