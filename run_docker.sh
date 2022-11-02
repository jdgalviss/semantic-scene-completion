#!/bin/bash
# sudo xhost +si:localuser:root
# XSOCK=/tmp/.X11-unix

sudo docker run -it --rm  \
    --net=host  \
    --privileged \
    --runtime=nvidia \
    -v `pwd`/semantic-scene-completion:/usr/src/app/semantic-scene-completion \
    -v /home/galvis/data/semantickitti/dataset/dataset/:/usr/src/app/data \
    --shm-size 32G \
    ssc "$@"


    # -e DISPLAY=$DISPLAY \
    # -v $XSOCK:$XSOCK \
    # -v $HOME/.Xauthority:/root/.Xauthority \