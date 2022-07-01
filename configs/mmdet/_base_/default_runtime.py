checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(
        #     type='MlflowLoggerHook',
        #     exp_name='thermals',
        #     tags=dict()
        # ),
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# Apparently this is discouraged and we should just use the evaluation flag with a
# validation pipeline
# workflow = [('train', 1), ('val', 1)]

# Will be placed within the data_root
mlflow_tracking_uri = 'sqlite:////path/to/mlruns.db'
mlflow_artifact_root = '/path/to/mlartifacts/'
