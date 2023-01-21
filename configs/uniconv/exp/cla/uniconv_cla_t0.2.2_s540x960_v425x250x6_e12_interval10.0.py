# -*- coding: utf-8 -*-
_base_ = 'uniconv_cla_v0.2.2_s540x960_v425x250x6_e40_interval1.0.py'
class_names = [
    'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'
]

total_epochs = 12
load_interval_train = 10
load_interval_test = 10

data = dict(
    train=dict(
        dataset=dict(
            load_interval=load_interval_train
        )
    ),
    val=dict(
        load_interval=load_interval_test
    ),
    test=dict(
        load_interval=load_interval_test
    )
)
