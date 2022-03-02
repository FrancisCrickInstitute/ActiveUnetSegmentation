#!/bin/bash

create_model -c boot-unet.json
attach_data_sources -c boot-unet.json
inspect_data_sources -c boot-unet.json
train_model -c boot-unet.json
predict_image boot-unet-latest.h5 sample.tif
