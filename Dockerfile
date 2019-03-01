FROM alpine:3.9

COPY . /root/pivot-schedule

RUN apk add --no-cache --update python3 py3-pip gcc python3-dev musl-dev freetype-dev libpng-dev \
    && pip3 install --upgrade --no-cache-dir pip \
    && cd /root/pivot-schedule \
    && pip3 install --no-cache-dir -r /root/pivot-schedule/requirements.txt \
    && rm -rf /root/pivot-schedule/alibaba/jobs/* \
    && mkdir -p /jobs /output

ENV PYTHONPATH /root/pivot-schedule
ENV JOB_DIR /jobs
ENV OUTPUT_DIR /output

ENTRYPOINT ["/usr/bin/python3", "/root/pivot-schedule/alibaba/sim.py"]
