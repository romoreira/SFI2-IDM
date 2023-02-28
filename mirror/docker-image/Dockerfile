FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
    net-tools \
    curl \
    tcpdump \
    python3 \
    python3-pip
RUN pip3 install Flask
COPY index.html /home
COPY server.py /home
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
