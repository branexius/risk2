import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import requests
import time

import math
# import plotly.graph_objects as go
from scipy.optimize import curve_fit
import sys
import numpy.polynomial.polynomial as poly
from pycoingecko import CoinGeckoAPI
from datetime import datetime
from datetime import date
from matplotlib import ticker

start_date = "01/01/2006"
end_date = "31/12/2021"
#end_date = str(date.today().strftime("%d/%m/%Y"))

date_format = "%d/%m/%Y"
z = datetime.strptime(start_date, date_format)
k = datetime.strptime(end_date, date_format)
zacetek = int(z.timestamp())
konec = int(k.timestamp())

ticker0 = "bitcoin"
vs = "usd"
API_KEY = '1sTj17l7Y0lZtMod7qtznlbmbyX'


def glassnode_price():
    res = requests.get('https://api.glassnode.com/v1/metrics/market/price_usd_close',
                       params={'a': "btc",
                               's': zacetek,
                               'u': konec,
                               'api_key': API_KEY})

    global data
    data = pd.read_json(res.text, convert_dates=['t'])
    #data.set_index("t", inplace=True)
    data.interpolate(method='cubic', inplace=True)


    #glassnode_NUPL()
    glassnode_VOL()
    #glassnode_nrpl()
    #glassnode_price1()
    calculate_risk()




def glassnode_price1():
    res = requests.get('https://api.glassnode.com/v1/metrics/market/price_usd_close',
                       params={'a': "btc",
                               's': zacetek,
                               'u': konec,
                               'api_key': API_KEY})

    global data12
    data12 = pd.read_json(res.text, convert_dates=['t'])
    #data.set_index("t", inplace=True)
    data12.interpolate(method='cubic', inplace=True)

    #print("glassnodetime:",data12["t"])
    #print(data12)


def glassnode_NUPL():
    res = requests.get('https://api.glassnode.com/v1/metrics/indicators/net_unrealized_profit_loss',
                       params={'a': ticker0,
                               's': zacetek,
                               'u': konec,
                               'api_key': API_KEY})
    global nupl
    global nupl0
    nupl = pd.read_json(res.text, convert_dates=['t'])
    # nupl.set_index("t", inplace=True)
    nupl0_max = nupl.max()
    nupl0_min = nupl.min()
    nupl0 = (nupl - nupl0_min) / (nupl0_max - nupl0_min)
    # print(nupl)
    # nupl0.loc[nupl['v'] < 0.5] *= (-1)


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


def glassnode_nrpl():
    res = requests.get("https://api.glassnode.com/v1/metrics/market/marketcap_usd",
                       params={'a': ticker0,
                               's': zacetek,
                               'u': konec,
                               'api_key': API_KEY})
    global mkc
    global mkc0
    mkc = pd.read_json(res.text, convert_dates=['t'])
    # mkc.set_index("t", inplace=True)
    # mkc = np.log(mkc)
    max = mkc.max()
    min = mkc.min()
    mkc0 = (mkc - min) / (max - min)


def calculate_risk():
    global risk
    global risk0
    global MA
    global risk55
    global MAD
    global BMS
    global MA0
    global RS
    global vola
    global log_reg
    global OBV
    global cow
    global volu
    global log_reg69
    global btp0, btp, btp1, btp2, btp3, btpstd, btpbtpstd
    global log_reg_min, log_reg_max
    '''
    sma_mala = 200
    sma_velika = 100
    sma_mad = 20
    sma_cow = 20
    volvar = 20
    RSI = 20
    volatilnost = 20
    sma_risk55 = 200

    '''
    sma_mala = 50
    sma_velika = 140
    sma_mad = 350
    sma_cow = 50
    volvar = 200
    RSI = 50
    volatilnost = 50
    sma_risk55 = 150
    btp_var = 1460

    '''
    sma_mala = 20
    sma_velika = 200
    sma_mad = 140
    sma_cow = 20
    volume = 50
    RSI = 200
    volatilnost = 20
    sma_risk55 = 20
    '''

    def func(x, p1, p2):
        return p1 * np.log(x) + p2

    data1 = data
    # print(data1)
    data1 = data1[data1["v"] > 0]
    #data1 = data1.iloc[::-1]
    xdata = np.array([x + 1 for x in range(len(data1))])
    # print(xdata)
    ydata = np.log(data1["v"])
    # print(ydata)

    popt, pcov = curve_fit(func, xdata, ydata, p0=(3.0, -10))

    log_reg = func(xdata, popt[0], popt[1])
    # log_reg = func(xdata, 2.9, -19.4)
    # print(log_reg)
    log_reg = np.exp(log_reg)


    #log_reg0 = np.exp(log_reg)
    #log_reg0 = np.log(data["v"]) / log_reg0
    #log_reg0_max = log_reg0.max()
    #log_reg0_min = log_reg0.min()
    #log_reg0 = (log_reg0 - log_reg0_min) / (log_reg0_max - log_reg0_min)

    #log_reg0 = data["v"].rolling(span=1460, adjust=False).mean()

    # 4.5 - log_reg69
    #x = np.array([x + 1 for x in range(len(data))])

    #log_reg69=1/(log_reg69)

    def sma(x):
        sma = data["v"].rolling(x, min_periods=1).mean()
        return sma

    def ema(x):
        ema = data["v"].ewm(span=x, min_periods=1, adjust=False).mean()
        return ema

    def normal(z):
        return (z-z.min())/(z.max()-z.min())



    log_reg_min = 4.4 * np.log(xdata) - 27.5
    #log_reg_min = 7 * np.log(xdata) - 48
    log_reg_min = np.exp(log_reg_min)

    log_reg_max = 3.5 * np.log(xdata) - 17.8
    log_reg_max = np.exp(log_reg_max)


    MA = (sma(sma_mala) / sma(sma_velika))
    MA = normal(MA)

    '''
    MA_max = MA.max()
    MA_min = MA.min()
    MA = (MA - MA_min) / (MA_max - MA_min)
    print(MA)
    '''
    '''
    # print(MA)
    # MAD = np.log(MA)

    # MAD.fillna(0, inplace=True)
    MAD = sma(sma_mad).diff()
    MAD_max = MAD.max()
    MAD_min = MAD.min()
    MAD = (MAD - MAD_min) / (MAD_max - MAD_min)
    '''
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
    print (cow)
    '''
    '''
    vol = vol0["v"].rolling(volvar).mean()
    vol0_max = vol.max()
    vol0_min = vol.min()
    volu = (vol - vol0_min) / (vol0_max - vol0_min)
    # print(vol)
    '''
    # 2.5
    '''
    force = vol0["v"].rolling(volvar).mean() * data["v"]
    force = np.log(force)
    force_max = force.max()
    force_min = force.min()
    force = (force - force_min) / (force_max - force_min)
    '''
    '''
    # 2.75
    OBV = []
    OBV.append(0)
    for i in range(1, len(data["v"])):
        if data["v"][i] > data["v"][i-1]:  # If the closing price is above the prior close price
            OBV.append(OBV[-1] + vol0["v"][i])  # then: Current OBV = Previous OBV + Current Volume
        elif data["v"][i] < data["v"][i-1]:
            OBV.append(OBV[-1] - vol0["v"][i])
        else:
            OBV.append(OBV[-1])
    print(OBV)

    OBV = pd.Series(OBV)
    OBV = OBV.ewm(span=20, adjust=False).mean()
    #OBV = np.log(OBV)
    OBV_max = OBV.max()
    OBV_min = OBV.min()
    OBV = (OBV - OBV_min) / (OBV_max - OBV_min)
    print(OBV)
    '''

    n = RSI
    delta = data["v"].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(window=n, min_periods=1).mean()
    RolDown = dDown.rolling(window=n, min_periods=1).mean().abs()

    RS = RolUp / RolDown
    RS = np.log(RS)
    RS_max = RS.max()
    RS_min = RS.min()
    RS = (RS - RS_min) / (RS_max-RS_min)


    '''
    r = volatilnost
    volat = np.log(data["v"] / data["v"].shift(-1))
    volat.fillna(0, inplace=True)
    volat = volat.rolling(window=r).std(ddof=0) * np.sqrt(r)

    BMS_max = volat.max()
    BMS_min = volat.min()
    vola = (volat - BMS_min) / (BMS_max - BMS_min)
    print(vola)
    '''


    risk55 = sma(sma_risk55)
    risk55 = (risk55 / data["v"]).diff()

    #orig
    #risk55 = risk55.diff()/data["v"]


    # risk55.fillna(0, inplace=True)
    risk55_max = risk55.max()

    risk55_min = risk55.min()

    risk55 = (risk55 - risk55_min) / (risk55_max - risk55_min)

    # risk55 = np.log(risk55)
    # pd.set_option("display.max_rows", 1000)
    # np.set_printoptions(threshold=sys.maxsize)

    #log_reg69 = 11.453 + (-1.067 * log(x))
    '''
    #xdata = data["t"].dropna()
    #print(xdata.size)
    x = np.arange(1, data["t"].size+1)
    y = np.array(data["v"])
    print(data)
    print(x)
    print(y)
    #print(data["v"])
    fit = poly.polyfit(np.log(x), y, 1)
    print(fit)
    pf0 = fit[0]
    pf1=fit[1]

    log_reg = pf0 + pf1*(np.log(x))
    print(log_reg)
    '''
    btpmean = data["v"].rolling(1460, min_periods=0).mean()
    btpstd = data["v"].rolling(window=1460, min_periods=0).std()
    btp0 = btpmean
    btp1 = btpmean + 2*btpstd
    btp2 = btpmean + 6*btpstd
    btp3 = btpmean + 8*btpstd
    #np.set_printoptions(threshold=sys.maxsize)
    btpbtpstd = (np.array(data["v"]) - btp0) / (btp3)


    btpbtpstd_max = btpbtpstd.max()
    btpbtpstd_min = btpbtpstd.min()
    btpbtpstd = (btpbtpstd - btpbtpstd_min) / (btpbtpstd_max - btpbtpstd_min)
    print(btpbtpstd)
    #log_reg69 = normal(log_reg69)

    btp = (np.array(data["v"]) - log_reg_min)/(log_reg_max)

    btp = (btp-1*btp.min())/(btp.max()-1*btp.min())

    risk = (btp)

    #risk = risk.rolling(window=7).mean()
    risk_max = risk.max()
    risk_min = risk.min()
    risk = (risk - risk_min) / (risk_max - risk_min)
    #print("risk:",risk)
    # risk = nupl+ 0.1*risk2 + 0.2*risk3 + 0.3*risk4 + 0.4*risk5

    plot_the_shit()


def plot_the_shit():
    color1 = "k"
    color2 = "r"
    color3 = "c"
    color4 = "b"
    color5 = "g"
    color6 = "m"
    color7 = "k"
    color8 = "w"

    fig, ax1 = plt.subplots(figsize=(16, 9))
    plt.subplots_adjust(left=0.04, right=0.95, top=0.95)
    plt.xticks(rotation=45)

    plt.figtext(0.45, 0.97, 'Grande minus chart', size=15, color='black')
    plt.figtext(0.92, 0.9, '5x', size=15, color='black')
    plt.figtext(0.92, 0.81, '4x', size=15, color='black')
    plt.figtext(0.92, 0.73, '3x', size=15, color='black')
    plt.figtext(0.92, 0.64, '2x', size=15, color='black')
    plt.figtext(0.92, 0.56, '1x', size=15, color='black')
    plt.figtext(0.92, 0.48, '-1x', size=15, color='black')
    plt.figtext(0.92, 0.39, '-2x', size=15, color='black')
    plt.figtext(0.92, 0.31, '-3x', size=15, color='black')
    plt.figtext(0.92, 0.22, '-4x', size=15, color='black')
    plt.figtext(0.92, 0.14, '-5x', size=15, color='black')
    plt.figtext(0.01, 0.02, 'Property of Minusarji. Not financial advice!', size=10, color='black')

    # plt.style.use("dark_background")

    ax1.semilogy(data["t"], data["v"], zorder=10, label='Price' + " " + ticker0 + " " + "/" + " " + vs, color=color1)
    ax1.set_ylim(0)
    ax1.plot(data["t"], btp3, zorder=2, label='top', color=color3)
    ax1.plot(data["t"], btp0, zorder=1, label='bottom', color=color5)
    ax1.plot(data["t"], btp1, zorder=1, label='fair value', color=color6)
    #ax1.plot(data["t"], log_reg_max, zorder=1, label='btpsd', color=color7)
    #ax1.plot(data["t"], btpbtpstd, zorder=1, label='stfd', color=color8)

    ax2 = ax1.twinx()
    #ax2.plot(data["t"], risk, zorder=5, label='Risk', color=color2)
    ax2.plot(data["t"], btpbtpstd, zorder=1, label='stfd', color=color4)


    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    ax1.set_ylabel('exp cene $', color=color1)
    ax1.tick_params(axis='y', color=color1)

    fmt_half_year = mdates.MonthLocator(bymonth=(1, 4, 7, 10))
    ax1.xaxis.set_major_locator(fmt_half_year)

    #print(ticker)
    #ax1.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    fmt_month = mdates.MonthLocator()
    ax1.xaxis.set_minor_locator(fmt_month)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax1.xaxis.set_tick_params(width=3)
    '''
    my_year_month_fmt = mdates.DateFormatter('%m-%y')
    ax1.xaxis.set_major_formatter(my_year_month_fmt)
    delta_dnevi = ((k.year - z.year)*12 + (k.month - z.month))
    print(delta_dnevi)

    #y_obdobje = delta_dnevi.days/30.5
    ax1.xaxis.set_major_locator(plt.MaxNLocator(delta_dnevi/4))
    ax1.set_xlabel("t")
    #for label in ax1.xaxis.get_ticklabels()[::2]:
        #label.set_visible(False)
    '''

    ax2.set_ylabel("risk", color=color2)
    ax2.set_ylim([0, 1])
    ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax2.tick_params(axis='y', color=color2)

    # ax3.set_ylabel("NUPL", color=color3)
    # ax3.yaxis.set_major_locator(plt.MaxNLocator(10))
    # ax3.tick_params(axis='y', color=color3, labelsize=30)
    # ax3.spines['right'].set_position(('outward', 50))

    '''
    ax4.legend(loc='best')
    ax4.set_ylabel("stfd", color=color4)
    ax4.tick_params(axis='y', color=color4)
    #ax4.yaxis.set_major_locator(ticker.Logformatter(4, integer = True))
    ax4.spines['right'].set_position(('outward', 100))
    '''

    # ax1.axvspan(8, 14, alpha=0.5, color='red')

    ax2.axhspan(0, 0.1, 0, 1, facecolor="limegreen", alpha=1)
    ax2.axhspan(0.1, 0.2, 0, 1, facecolor="limegreen", alpha=0.8)
    ax2.axhspan(0.2, 0.3, 0, 1, facecolor="limegreen", alpha=0.6)
    ax2.axhspan(0.3, 0.4, 0, 1, facecolor="limegreen", alpha=0.4)
    ax2.axhspan(0.4, 0.5, 0, 1, facecolor="limegreen", alpha=0.2)
    ax2.axhspan(0.5, 0.6, 0, 1, facecolor="gold", alpha=0.2)
    ax2.axhspan(0.6, 0.7, 0, 1, facecolor="gold", alpha=0.4)
    ax2.axhspan(0.7, 0.8, 0, 1, facecolor="gold", alpha=0.6)
    ax2.axhspan(0.8, 0.9, 0, 1, facecolor="gold", alpha=0.8)
    ax2.axhspan(0.9, 1, 0, 1, facecolor="gold", alpha=1)

    ax2.grid(axis='y')
    ax1.grid(axis='x')
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.show()

glassnode_price()