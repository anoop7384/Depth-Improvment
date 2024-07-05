Introduction
This guide will help you run the provided Python script, which captures an image from various sources (webcam, local directory, IP camera), processes it for depth map improvement.

Prerequisites
Before running the script, ensure you have the following installed:

1. Python 3.6+

2. Install all packages :
    run command : pip install -r requirements.txt



Starting the Milvus Database Server

NOTE: Linux system is required to start this Server

Make sure docker-compose.yml file present in current directory

    run command : sudo docker-compose up -d


Running the Python script

    run command : python depthmap.py




