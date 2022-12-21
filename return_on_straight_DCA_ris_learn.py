import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import requests
import time
import datetime
import math
import itertools
from datetime import datetime as datet
from pycoingecko import CoinGeckoAPI
from datetime import datetime
from datetime import date
from matplotlib import ticker



if __name__ == "__main__":
    start_date = "01/01/2015"
    end_date = "16/12/2021"
    na = 7
    vlozek0 = 50
else:
    import risk_learn_v2_numpy_clean
    start_date = risk_learn_v2_numpy_clean.start_date
    end_date = risk_learn_v2_numpy_clean.end_date
    na = risk_learn_v2_numpy_clean.na
    vlozek0 = risk_learn_v2_numpy_clean.vlozek0

zacetek = int(datetime.strptime(start_date, "%d/%m/%Y").timestamp())
konec = int(datetime.strptime(end_date, "%d/%m/%Y").timestamp())

ticker0 = "bitcoin"
vs = "usd"
API_KEY = '1sTj17l7Y0lZtMod7qtznlbmbyX'


def glassnode_price():
    global data
    global data0
    global data1
    global price

    def unix_to_datetime(unix_time):
        '''unix_to_datetime(1622505700)=> ''2021-06-01 12:01am'''
        ts = int(unix_time / 1000 if len(str(unix_time)) > 10 else unix_time)  # /1000 handles milliseconds
        return np.datetime64(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d').lower())

    cg = CoinGeckoAPI()  # Retrieve Bitcoin data in USD
    result = cg.get_coin_market_chart_range_by_id(
        id=ticker0,
        vs_currency=vs,
        from_timestamp=zacetek,
        to_timestamp=konec
    )
    time = [unix_to_datetime(i[0]) for i in result['prices']]

    p_array = np.array(result['prices'])
    price = p_array[:, 1]

    data = pd.DataFrame({'t': time, 'v': price})
    data.interpolate(method='cubic', inplace=True)
    #print("geckotime:", data["t"])
    data0 = np.log10(data["v"])
    data1 = data






'''
def glassnode_price():
    res = requests.get('https://api.glassnode.com/v1/metrics/market/price_usd_close',
                       params={'a': ticker,
                               's': zacetek,
                               'u': konec,
                               'api_key': API_KEY})
    global data
    global data0
    global data1
    data = pd.read_json(res.text, convert_dates=['t'])
    #data.set_index("t", inplace=True)
    data.interpolate(method='cubic', inplace=True)
    data0 = np.log10(data["v"])
    data1 = data
    print(data1)

    glassnode_NUPL()
    glassnode_VOL()
    glassnode_nrpl()
    calculate_risk()
'''


fond0 = 0
stBTC = 0
glassnode_price()
data0 = price[::na]
#print(data0)
stBTClist = []
fondlist = []
maxi = []
indie = []

for i in data0:
    #print(i)
    vlozek = vlozek0
    fond0 = fond0 - vlozek
    stBTC = stBTC + (vlozek / i)

    #print(data.index)
    #print("nakup_BTC5:", stBTC)
    #print(len(data0) - 1)
    #print(j/7)
    #print(data["v"][j])

    if i == data0[-1]:
        #print(data["v"][j])
        stBTClist.append(stBTC * i)
        fondlist.append(fond0)


zipped = [x + y for x, y in zip(stBTClist, fondlist)]
stBTClist = round(float(("".join(str(i)for i in stBTClist))), 2)
fondlist = round(float(("".join(str(i)for i in fondlist))), 2)
zipped = round(float(("".join(str(i)for i in zipped))), 2)

print("STRAIGHT DCA")
print("Obdobje:", "od", start_date, "do", end_date)
print("Value of inv.:", stBTClist)
print("Cost of inv.:", fondlist)
print("Delta of inv.:", zipped)
ROI = (stBTClist - abs(fondlist))/abs(fondlist)
print("ROI:",round(ROI, 2))
