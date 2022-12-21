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
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from pycoingecko import CoinGeckoAPI
from datetime import datetime
from datetime import date
from matplotlib import ticker



start_date = "01/08/2015"
end_date = "16/12/2021"

zacetek = int(datetime.strptime(start_date, "%d/%m/%Y").timestamp())
konec = int(datetime.strptime(end_date, "%d/%m/%Y").timestamp())

ticker0 = "bitcoin"
vs = "usd"
API_KEY = '1sTj17l7Y0lZtMod7qtznlbmbyX'


def glassnode_price():
    global data
    global data0
    global data1

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

    glassnode_VOL()




def glassnode_VOL():
    global vol0


    def unix_to_datetime(unix_time):
        '''unix_to_datetime(1622505700)=> ''2021-06-01 12:01am'''
        ts = int(unix_time / 1000 if len(str(unix_time)) > 10 else unix_time)  # /1000 handles milliseconds
        return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d').lower()

    cg = CoinGeckoAPI()  # Retrieve Bitcoin data in USD
    result = cg.get_coin_market_chart_range_by_id(
        id=ticker0,
        vs_currency=vs,
        from_timestamp=zacetek,
        to_timestamp=konec
    )
    time = [unix_to_datetime(i[0]) for i in result['prices']]

    v_array = np.array(result['total_volumes'])
    volume = v_array[:, 1]

    vol0 = pd.DataFrame({'t': time, 'v': volume})
    vol0.interpolate(method='cubic', inplace=True)
    #print("coingeckovol:", vol0)


    res = requests.get('https://api.glassnode.com/v1/metrics/transactions/transfers_volume_sum',
                       params={'a': "btc",
                               's': zacetek,
                               'u': konec,
                               'api_key': API_KEY})

    vol = pd.read_json(res.text, convert_dates=['t'])
    # vol.set_index("t", inplace=True)
    #print("glassnodevol:", vol)
    #vol0 = vol

    # stfd = np.log(stfd)

    # stfd0.loc[stfd0['v'] < 0.5] *=(-1)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(stfd0.all)



glassnode_price()
stBTClist = []
fondlist = []
maxi = []
indie = []



def calculate_risk(sma_mala, sma_velika, sma_MAD, volvar, RSI, sma_risk55):
    global risk
    global risk0
    global MA
    global risk55
    global MAD
    global BMS
    global MA0
    global RS
    global v
    global vol
    global cow


    def sma(x):
        sma = data["v"].rolling(window=x).mean()
        return sma

    def ema(y):
        ema = data["v"].ewm(span=y, adjust=False).mean()
        return ema

    #0
    MA = sma(sma_mala)/(sma(sma_velika))

    MA_max = MA.max()
    MA_min = MA.min()
    MA0 = (MA - MA_min) / (MA_max - MA_min)

    #1

    MAD = sma(sma_MAD).diff()
    MAD_max = MAD.max()
    MAD_min = MAD.min()
    MAD = (MAD - MAD_min) / (MAD_max - MAD_min)

    '''
    #cowen RNG BS
    korcas=[]
    for i in data.index.tolist():
        delta = data.index[i]
        cas = 1/(math.sqrt(delta+1))
        korcas.append(cas)

    cow = ((4*math.pi)**2) * sma(sma_cow) * korcas
    #cow = np.log(cow)
    cow_max = cow.max()
    cow_min = cow.min()
    cow = (cow - cow_min) / (cow_max - cow_min)
    '''
    '''
    #2
    vol = vol0["v"].rolling(volvar).mean()
    vol0_max = vol.max()
    vol0_min = vol.min()
    vol = (vol - vol0_min) / (vol0_max - vol0_min)
    '''

    #2.5
    force = vol0["v"].rolling(volvar).mean() * data["v"]
    force = np.log(force)
    force_max = force.max()
    force_min = force.min()
    force = (force - force_min) / (force_max - force_min)


    #3
    n = RSI
    delta = data["v"].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    RolUp = dUp.rolling(window=n).mean()
    RolDown = dDown.rolling(window=n).mean().abs()
    RS = RolUp / RolDown
    RS = np.log(RS)
    RS_max = RS.max()
    RS_min = RS.min()
    RS = (RS - RS_min) / (RS_max - RS_min)

    def func(x, p1, p2):
        return p1 * np.log(x) + p2

    data1 = data
    #print(data1)
    data1 = data1[data1["v"] > 0]
    # data1 = data1.iloc[::-1]
    xdata = np.array([x + 1 for x in range(len(data1))])
    #print(xdata)
    ydata = np.log(data1["v"])
    # print(ydata)

    popt, pcov = curve_fit(func, xdata, ydata, p0=(3.0, -10))
    #print(popt)
    log_reg = func(xdata, popt[0], popt[1])
    #print(log_reg)
    log_reg0 = np.exp(log_reg)
    log_reg0 = np.log(data["v"])/log_reg0
    log_reg0_max = log_reg0.max()
    log_reg0_min = log_reg0.min()
    log_reg0 = (log_reg0 - log_reg0_min) / (log_reg0_max - log_reg0_min)
    '''
    #4
    r = volatilnost
    volat = np.log(data["v"]/data["v"].shift(-1))
    volat.fillna(0, inplace=True)
    volat = volat.rolling(window=r).std(ddof=0)*np.sqrt(r)
    BMS_max = volat.max()
    BMS_min = volat.min()
    v = (volat - BMS_min) / (BMS_max - BMS_min)
    '''

    #5
    risk55=sma(sma_risk55)
    risk55 = (data["v"] / risk55).diff()
    risk55_max = risk55.max()
    risk55_min = risk55.min()
    risk55 = (risk55 - risk55_min) / (risk55_max - risk55_min)


    risk = (MA0 + MAD + risk55 - log_reg0 + RS + force)
    risk = risk.rolling(window=7).mean()
    risk_max = risk.max()
    risk_min = risk.min()
    risk = (risk - risk_min) / (risk_max - risk_min)
    print(risk)
    risk_opti(risk)


def risk_opti(risk):
    fond0 = 0
    vlozek0 = 50
    stBTC = 0
    na=7
    risk0 = risk.iloc[::na]
    #print(risk0)
    data0 = data.iloc[::na]
    #print(risk0)
    #print(data)
    #pd.set_option("display.max_rows", 1000)
    #print(data0)
    #print(data["v"][j])
    #print(len(risk))
    #print(risk.fillna(0))
    risknp = risk0.to_numpy()

    if len(risk0.dropna()) == 0:
        A=0
    else:
        #for j,i in enumerate(risk0):
        for j,i in risk0.iteritems():
            #print("asd",j)
            #print("dsa",i)
        #for j, i in risk.fillna(0).values:
            if 0 < i <= 0.1:
                print(i,j)
                y = 5
                vlozek = vlozek0 * y
                fond0 = fond0 - vlozek
                print("nakup_fond5:", fond0)
                print("nakup_cenaBTC:", data["v"][j])
                stBTC = stBTC + (vlozek / data["v"][j])
                print("nakup_BTC5:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*(data["v"][j]))
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(len(risk0)-1)

                #return fond0, stBTC
                #return


            elif 0.1 < i <= 0.2:
                print(i, j)
                y = 4
                vlozek = vlozek0 * y
                fond0 = fond0 - vlozek
                print("nakup_fond4:", fond0)
                print("nakup_cenaBTC:", data["v"][j])
                stBTC = stBTC + (vlozek / data["v"][j])
                print("nakup_BTC4:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*(data["v"][j]))
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            elif 0.2 < i <= 0.3:
                print(i, j)
                y = 3
                vlozek = vlozek0 * y
                fond0 = fond0 - vlozek
                print("nakup_fond3:", fond0)
                print("nakup_cenaBTC:", data["v"][j])
                stBTC = stBTC + (vlozek / data["v"][j])
                print("nakup_BTC3:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            elif 0.3 < i <= 0.4:
                print(i, j)
                y = 2
                vlozek = vlozek0 * y
                fond0 = fond0 - vlozek
                print("nakup_fond2:", fond0)
                print("nakup_cenaBTC:", data["v"][j])
                stBTC = stBTC + (vlozek / data["v"][j])
                print("nakup_BTC2:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            elif 0.4 < i <= 0.5:
                print(i, j)
                y = 1
                vlozek = vlozek0 * y
                fond0 = fond0 - vlozek
                print("nakup_fond1:", fond0)
                print("nakup_cenaBTC:", data["v"][j])
                stBTC = stBTC + (vlozek / data["v"][j])
                print("nakup_BTC1:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            elif 0.5 < i <= 0.6:
                print(i, j)
                y = 1
                vlozek = vlozek0 * y
                fond0 = fond0 + vlozek
                print("prodaja_fond1:", fond0)
                print("prodaja_cenaBTC:", data["v"][j])
                stBTC = stBTC - (vlozek / data["v"][j])
                # print(data["v"][j])

                print("prodaja_BTC1:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(j)
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            elif 0.6 < i <= 0.7:
                print(i, j)
                y = 2
                vlozek = vlozek0 * y
                fond0 = fond0 + vlozek
                print("prodaja_fond2:", fond0)
                print("prodaja_cenaBTC:", data["v"][j])
                stBTC = stBTC - (vlozek / data["v"][j])
                # print(data["v"][j])

                print("prodaja_BTC2:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(j)
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            elif 0.7 < i <= 0.8:
                print(i, j)
                y = 3
                vlozek = vlozek0 * y
                fond0 = fond0 + vlozek
                print("prodaja_fond3:", fond0)
                print("prodaja_cenaBTC:", data["v"][j])
                stBTC = stBTC - (vlozek / data["v"][j])
                # print(data["v"][j])

                print("prodaja_BTC3:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(j)
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            elif 0.8 < i <= 0.9:
                print(i, j)
                y = 4
                vlozek = vlozek0 * y

                fond0 = fond0 + vlozek
                print("prodaja_fond4:", fond0)
                print("prodaja_cenaBTC:", data["v"][j])
                stBTC = stBTC - (vlozek / data["v"][j])
                # print(data["v"][j])

                print("prodaja_BTC4:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga",stBTC*data["v"][j])
                    #print(j)
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            elif 0.9 < i <= 1:
                print(i, j)
                y = 5
                vlozek = vlozek0 * y
                #print(vlozek)
                fond0 = fond0 + vlozek
                print("prodaja_fond5:", fond0)
                print("prodaja_cenaBTC:", data["v"][j])
                stBTC = stBTC - (vlozek / data["v"][j])
                # print(data["v"][j])

                print("prodaja_BTC5:", stBTC)
                print(".........................................................")
                if j/na == (len(risk0)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga", stBTC*data["v"][j])
                    #print(j)
                    #print(len(risk0) - 1)
                #return fond0, stBTC
                #return

            else:
                print(i, j)
                vlozek = vlozek0
                fond0 = fond0
                print("prodaja_fond0:", fond0)
                print("prodaja_cenaBTC:", data["v"][j])
                stBTC = stBTC
                # print(data["v"][j])

                print("prodaja_BTC0:", stBTC)
                print(".........................................................")
                if j/na == (len(risk)-1):
                    #print(j)
                    stBTClist.append(stBTC*data["v"][j])
                    fondlist.append(fond0)
                    #print("fond", fond0)
                    #print("druga", stBTC*data["v"][j])
                #return fond0, stBTC
                #return

        #print(stBTClist, fondlist)

        zipped=[x + y for x, y in zip(stBTClist, fondlist)]
        #print("ipped:", zipped)
        maximual = max(zipped)
        print(maximual)
        maxi.append(maximual)
        print("max:", maximual)


#parametri = [sma_mala, sma_velika, sma_MAD,  sma_risk55]
# sprem = np.arange(50, 130, 10).tolist()
sprem = [20, 50, 50, 50, 70, 70]
#sprem = [20, 200, 140, 200, 20]
#value = [p for p in itertools.product(sprem, repeat=len(parametri))]
# print(value[22980])

calculate_risk(20, 50, 50, 50, 70, 70)
#calculate_risk(20, 200, 140, 200, 20)

print(max(maxi))

startTime = datet.now()
glassnode_price()
print(datet.now() - startTime)
print("parametri:",sprem)

print("Dynamic DCA")
print("Obdobje:", "od", start_date, "do", end_date)
print("Value of inv.:", stBTClist)
print("Cost of inv.:", fondlist)
print("Delta of inv.:", max(maxi))
ROI = ((int(float(("".join(str(i)for i in stBTClist))))) + (int(float(("".join(str(i)for i in fondlist))))))/(-int(float(("".join(str(i)for i in fondlist)))))
print("ROI:",ROI)