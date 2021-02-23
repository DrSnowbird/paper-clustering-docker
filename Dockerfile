FROM openkbs/jdk-mvn-py3
MAINTAINER DrSnowbird@openkbs.org

COPY python/ $HOME/python
WORKDIR $HOME/python

RUN /usr/bin/python3 -m pip install --upgrade pip 

RUN pip3 install -r requirements.txt

EXPOSE 8180

RUN sudo chown -R $USER:$USER $HOME

RUN ls -al $HOME/*
CMD ["python3", "Clustering_Papers.py"]
