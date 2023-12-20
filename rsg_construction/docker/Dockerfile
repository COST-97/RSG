# syntax = edrevo/dockerfile-plus

INCLUDE+ ./Dockerfile.common

COPY ./requirements.txt /dep/requirements.txt

RUN pip install -r /dep/requirements.txt

CMD ["bash"]