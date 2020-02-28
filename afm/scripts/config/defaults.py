import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.ARCH = "atrous"
# _C.MODEL.WEIGHT  = ""


# --------------------------------------------
# INPUT
# --------------------------------------------

_C.INPUT = CN()
_C.INPUT.STRATEGY = 'fixed'
_C.INPUT.IN_RES = 320
_C.INPUT.OUT_RES = 320


# --------------------------------------------
# Dataset
# --------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST  = ()

# --------------------------------------------
# Dataloader
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
# --------------------------------------------

# --------------------------------------------
# Criteron
_C.CRITERION = CN()
_C.CRITERION.LOSS = "l1"
_C.CRITERION.REDUCTION = "mean"
# --------------------------------------------
# --------------------------------------------
# Solver
# --------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.NUM_EPOCHS = 200

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0002
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (160,)
_C.SOLVER.BATCH_SIZE = 16
_C.SOLVER.CHECKPOINT_PERIOD = 10

_C.save_dir = "."

_C.image_topic = ""

# --------------------------------------------
# Test configurations
# --------------------------------------------
_C.TEST = CN()
_C.TEST.TRANSFORM = True
_C.TEST.OUTPUT_MODE = "display"
_C.TEST.DISPLAY = CN()
_C.TEST.DISPLAY.THRESHOLD = 0.2

# --------------------------------------------
# camera intrinsic
# --------------------------------------------
_C.distortion_parameters=CN()
_C.distortion_parameters.k1= 0.0
_C.distortion_parameters.k2= 0.0
_C.distortion_parameters.p1= 0.0
_C.distortion_parameters.p2= 0.0

_C.projection_parameters=CN()
_C.projection_parameters.fx= 320.0
_C.projection_parameters.fy= 240.0
_C.projection_parameters.cx= 320.0
_C.projection_parameters.cy= 240.0

_C.width=CN()
_C.width=752
_C.height=CN()
_C.height=480
