#!/usr/bin/env bash
PORT=8086

PARAMS="-p ${PORT}:${PORT} --net=host --ipc=host -u $(id -u ${USER}):$(id -g ${USER})"
NAME="neural_textures"
HEAD_NAME="airuhead01:5000/${NAME}"
VOLUMES="-v /:/mounted"