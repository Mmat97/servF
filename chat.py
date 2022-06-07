import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import subprocess
from parlai.scripts.interactive import Interactive
from parlai.scripts.train_model import TrainModel
import warnings
import sys

warnings.filterwarnings("ignore")

emotion = sys.argv[1].lower()
model_root_fp = sys.argv[2]





blend_name = 'blender_empathetic_'
run_model = True

#print(model_root_fp)

if emotion.lower() == 'happy':
    model_fp = blend_name + 'happy/model'
elif emotion == 'fear':
    model_fp = blend_name + 'fear/model'
elif emotion == 'sad':
    model_fp = blend_name + 'sad/model'
elif emotion == 'surprise':
    model_fp = blend_name + 'surprise/model'
elif emotion == 'angry':
    model_fp = blend_name + 'anger/model'
else:
    run_model = False

if run_model:
    final_model_fp = model_root_fp + model_fp

    #print(final_model_fp)

    # call it with particular args
    print("eeeeeeeeetttttt")
    #Interactive.main(model_file=final_model_fp, single_turn=True)
    #Interactive.main(model_file='zoo:blender/blender_90M/model')
    Interactive.main(model_file='zoo:tutorial_transformer_generator/model')
    
else:
    print('Not a valid emotion...')