# -*- coding: utf-8 -*-

from .config import ( MODEL_FILE, MODEL_OUT, INPUT_SHAPE, KERNEL_SHAPE, 
POOLING_SHAPE, LEARNING_RATE, EPOCHS,
OPTIMIZER, LOSS_FUNCTION, STRIDE, BATCH_SIZE, MULTI_GPU, 
DATA_SOURCES, N_LABELS, DEPTH, N_FILTERS, ACTIVATION, VALIDATION_FRACTION, 
SPATIAL_DROPOUT_RATE, NORMALIZE_SAMPLES, SAMPLES_TO_PREDICT )

DEFAULT=0
def echo(*statement, level=DEFAULT):
    """
        For messages that *should* be displayed, print statments are all considered
        debug information and a susceptible to be removed.
    """
    print(*statement)