base_path = '../configs/selfsup/_base_'
_base_ = [
    base_path + '/models/byol.py',
    '../boris/traffic_byol.py',
#    '../_base_/datasets/imagenet_byol.py',
    base_path + '/schedules/lars_coslr-200e_in1k.py',
    base_path + '/default_runtime.py',
]

# additional hooks
# interval for accumulate gradient, total 8*32*16(interval)=4096
update_interval = 1
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

# optimizer
optimizer = dict(
    type='LARS',
    lr=4.8,
    momentum=0.9,
    weight_decay=1e-6,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)':
        dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True)
    })
optimizer_config = dict(update_interval=update_interval)

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
