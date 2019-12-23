from datetime import datetime,timedelta
from collections import OrderedDict
import pandas as pd

dates=["1953-01-01","2018-12-01"]
start,end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
m=OrderedDict(((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys()
print(m)