import numpy as np
import typing
from typing import Any, Tuple

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import tensorflow_text as tf_text

def load_data(path):
  text = path.read_text(encoding='utf-8')

  lines = text.splitlines()
  pairs = [line.split('\t') for line in lines]

  context = np.array([context for target, context in pairs])
  target = np.array([target for target, context in pairs])

  return target, context

path_to_file = "/Users/kakao/Sources/prompt-engineering-pt/spa-eng/spa.txt"
target_raw, context_raw = load_data(path_to_file)
print(context_raw[-1])