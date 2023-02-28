FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
    net-tools \
    curl \
    tcpdump \
    python3 \
    python3-pip
RUN pip3 install Flask
RUN pip3 install cicflowmeter
COPY index.html /home
COPY server.py /home
COPY flow.py /usr/local/lib/python3.8/dist-packages/cicflowmeter/
COPY flow_session.py /usr/local/lib/python3.8/dist-packages/cicflowmeter/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
