import cython
import pandas as pd
import numpy as np
import datetime
import math
import itertools

from datetime import datetime as datet
from scipy.optimize import curve_fit
from pycoingecko import CoinGeckoAPI
from datetime import datetime

cdef list maxi = []
cdef list indie = []
cdef list index_of_params = []
cdef list stBTClist = []
cdef list fondlist = []
cdef int zap = 0

cdef int r
cdef list risk_var
cdef double fond0
cdef int vlozek0
cdef double vlozek
cdef double stBTC
cdef double cena
cdef double risk0


cdef double sma_mala = 0
cdef double sma_velika = 0
cdef double sma_MAD = 0
cdef double volvar = 0
cdef double RSI = 0
cdef double sma_risk55 = 0
cdef list parametri
cdef list sprem



startTime = datet.now()

def calculate_risk(sma_mala, sma_velika, sma_MAD, RSI, volvar, sma_risk55):
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
    global data2
    global vol2

    def sma(x):
        sma = data2["v"].rolling(window=x).mean()
        return sma

    def ema(y):
        ema = data2["v"].ewm(span=y, adjust=False).mean()
        return ema

    # 0
    MA = sma(sma_mala) / (sma(sma_velika))

    MA_max = MA.max()
    MA_min = MA.min()
    MA = (MA - MA_min) / (MA_max - MA_min)

    # 1

    MAD = sma(sma_MAD).diff()
    MAD_max = MAD.max()
    MAD_min = MAD.min()
    MAD = (MAD - MAD_min) / (MAD_max - MAD_min)

    '''
    #cowen RNG BS
    korcas=[]
    for k in data2.index.tolist():
        delta = data2.index[k]
        cas = 1/(math.sqrt(delta+1))
        korcas.append(cas)

    cow = ((4*math.pi)**2) * sma(sma_cow) * korcas
    #cow = np.log(cow)
    cow_max = cow.max()
    cow_min = cow.min()
    cow = (cow - cow_min) / (cow_max - cow_min)
    #print (cow)
    '''

    '''
    #2
    vol = vol0["v"].rolling(volvar).mean()
    vol0_max = vol.max()
    vol0_min = vol.min()
    vol = (vol - vol0_min) / (vol0_max - vol0_min)
    '''

    # 2.5

    force = vol2["v"].rolling(volvar).mean() * data2["v"]
    force = np.log(force)
    force_max = force.max()
    force_min = force.min()
    force = (force - force_min) / (force_max - force_min)


    # 3
    n = RSI
    delta = data2["v"].diff()
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

    data1 = data2
    data1 = data1[data1["v"] > 0]
    xdata = np.array([x + 1 for x in range(len(data1))])
    ydata = np.log(data1["v"])

    popt, pcov = curve_fit(func, xdata, ydata, p0=(3.0, -10))
    log_reg = func(xdata, popt[0], popt[1])
    log_reg0 = np.exp(log_reg)
    log_reg0 = np.log(data2["v"]) / log_reg0
    log_reg0_max = log_reg0.max()
    log_reg0_min = log_reg0.min()
    log_reg0 = (log_reg0 - log_reg0_min) / (log_reg0_max - log_reg0_min)
    '''
    #4
    r = volatilnost
    volat = np.log(data2["v"]/data2["v"].shift(-1))
    volat.fillna(0, inplace=True)
    volat = volat.rolling(window=r).std(ddof=0)*np.sqrt(r)
    BMS_max = volat.max()
    BMS_min = volat.min()
    v = (volat - BMS_min) / (BMS_max - BMS_min)
    '''

    # 5
    risk55 = sma(sma_risk55)
    risk55 = (data2["v"] / risk55).diff()
    risk55_max = risk55.max()
    risk55_min = risk55.min()
    risk55 = (risk55 - risk55_min) / (risk55_max - risk55_min)

    risk = (MA + MAD + risk55 + RS - log_reg0 + force)
    risk_max = risk.max()
    risk_min = risk.min()
    risk = (risk - risk_min) / (risk_max - risk_min)
    '''
    print("MA", MA)
    print("MAD", MAD)
    #print("force", force)
    print("RSI", RS)
    print("risk55", risk55)
    print("log_reg0", log_reg0)
    print("risk:", risk)
    '''


def risk_opti(risk):

    global vlozek0
    global fond0
    global stBTC
    global na
    global cena

    risk0 = risk.iloc[-1]
    cena = data2["v"].iloc[-1]
    if math.isnan(risk0):
        A = 0
    elif 0 < risk0 <= 0.1:
        y = 5
        vlozek = vlozek0 * y
        fond0 = fond0 - vlozek
        stBTC = stBTC + (vlozek / cena)

    elif 0.1 < risk0 <= 0.2:
        y = 4
        vlozek = vlozek0 * y
        fond0 = fond0 - vlozek
        stBTC = stBTC + (vlozek / cena)

    elif 0.2 < risk0 <= 0.3:
        y = 3
        vlozek = vlozek0 * y
        fond0 = fond0 - vlozek
        stBTC = stBTC + (vlozek / cena)

    elif 0.3 < risk0 <= 0.4:
        y = 2
        vlozek = vlozek0 * y
        fond0 = fond0 - vlozek
        stBTC = stBTC + (vlozek / cena)

    elif 0.4 < risk0 <= 0.5:
        y = 1
        vlozek = vlozek0 * y
        fond0 = fond0 - vlozek
        stBTC = stBTC + (vlozek / cena)

    elif 0.5 < risk0 <= 0.6:
        y = 1
        vlozek = vlozek0 * y
        fond0 = fond0 + vlozek
        stBTC = stBTC - (vlozek / cena)

    elif 0.6 < risk0 <= 0.7:
        y = 2
        vlozek = vlozek0 * y
        fond0 = fond0 + vlozek
        stBTC = stBTC - (vlozek / cena)

    elif 0.7 < risk0 <= 0.8:
        y = 3
        vlozek = vlozek0 * y
        fond0 = fond0 + vlozek
        stBTC = stBTC - (vlozek / cena)

    elif 0.8 < risk0 <= 0.9:
        y = 4
        vlozek = vlozek0 * y
        fond0 = fond0 + vlozek
        stBTC = stBTC - (vlozek / cena)

    elif 0.9 < risk0 <= 1:
        y = 5
        vlozek = vlozek0 * y
        fond0 = fond0 + vlozek
        stBTC = stBTC - (vlozek / cena)

    else:
        vlozek = vlozek0
        fond0 = fond0
        stBTC = stBTC




def risk_get(value):
    import learntest111
    data8 = learntest111.data8
    vol0 = learntest111.vol0
    print(data8)
    global zap
    global data2
    global vol2


    for risk_var in value:
        '''
        global vlozek0
        global fond0
        global stBTC
        global na
        global cena
        '''
        risk_var=list(risk_var)
        fond0 = 0
        vlozek0 = 50
        stBTC = 0
        na = 7
        cena = 0

        cikel1 = datet.now()

        if risk_var[0] == risk_var[1]:
            pass
        else:
            for r in data8["t"][::na]:
                data2 = data8[(data8['t'] < r)]
                vol2 = vol0[data8["t"] < r]
                #print(data2)
                #print(len(data2))

                if len(data2) < 151:
                    pass
                else:
                    calculate_risk(*risk_var)
                    if risk.isnull().all():
                        pass
                    else:
                        risk_opti(risk)

        stBTClist.append(stBTC * cena)
        fondlist.append(fond0)

        index_of_params.append(risk_var)

        zipped = [x + y for x, y in zip(stBTClist, fondlist)]

        maximual = max(zipped)
        print("maximual:",maximual)
        maxi.append(maximual)

        indeks = zipped.index(max(zipped))
        print("index:", indeks)
        indie.append(indeks)

        print("parametri:", index_of_params[max(indie)])
        print("obdelana:", risk_var)
        zap+=1
        print("zaporedna:", zap)
        print("cikel1:", datet.now() - cikel1)


#from learntest111 import sprem
parametri = [sma_mala, sma_velika, sma_MAD, RSI, volvar, sma_risk55]
sprem = [20, 50, 100, 200]

value = [p for p in itertools.product(sprem, repeat=len(parametri))]
risk_get(value)