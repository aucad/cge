import json
import re
from typing import Any

import numpy as np
import pandas as pd


class Utility:

    @staticmethod
    def read_dataset(dataset_path):
        df = pd.read_csv(dataset_path).fillna(0)
        attrs = [re.sub(r'\W+', '*', a)
                 for a in [col for col in df.columns]]
        return attrs, np.array(df)

    @staticmethod
    def write_result(fn, content):
        with open(fn, "w") as outfile:
            json.dump(content, outfile, indent=4)
        print('Wrote result to', fn, '\n')

    @staticmethod
    def sdiv(num, denom, fault: Any = '', mult=True):
        return fault if denom == 0 else ((100 if mult else 1) * num / denom)

    @staticmethod
    def log(label: str, value):
        print(f'{label} '.ljust(18, '-') + str(value))

    @staticmethod
    def logr(label, num, den):
        a, b, ratio = round(num, 0), round(den, 0), Utility.sdiv(num, den)
        return Utility.log(label, f'{a} of {b} - {ratio:.1f} %')
