import collections
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from service.heartDiseaseService import HeartDiseaseService

if __name__ == '__main__':

    service = HeartDiseaseService()
    service.run_model()
