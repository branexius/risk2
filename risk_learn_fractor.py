import datetime
import itertools
import warnings
import math
from datetime import datetime
from datetime import datetime as datet
import bottleneck as bn
import numpy as np
from pycoingecko import CoinGeckoAPI
from scipy.optimize import curve_fit

start_date = "01/01/2015"
end_date = "16/12/2021"
na = 1
vlozek0 = 50

if __name__ == "__main__":
    startTime = datet.now()

    zacetek = int(datetime.strptime(start_date, "%d/%m/%Y").timestamp())
    konec = int(datetime.strptime(end_date, "%d/%m/%Y").timestamp())

    ticker0 = "bitcoin"
    vs = "usd"

    def glassnode_price():
        global data
        global data0
        global price
        global timeprice

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
        timeprice1 = [unix_to_datetime(i[0]) for i in result['prices']]
        timeprice = np.array(timeprice1, dtype="datetime64")
        p_array = np.array(result['prices'])
        price = p_array[:, 1]

    def glassnode_VOL():
        global vol0
        global volume
        global timevol

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
        timevol1 = [unix_to_datetime(i[0]) for i in result['total_volumes']]
        timevol = np.array(timevol1, dtype="datetime64")
        v_array = np.array(result['total_volumes'])
        volume = v_array[:, 1]

    def calculate_risk(sma_mala, sma_velika, sma_MAD, volvar, RSI):
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
        global vol0

        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
        warnings.filterwarnings(action='ignore', message='invalid value encountered in true_divide')
        def sma(x):
            sma = bn.move_mean(price2, window=x)
            return sma
        def ema(y):
            ema = bn.move_mean(price2, window=y)
            return ema

        # 0
        MA = sma(sma_mala) / (sma(sma_velika))
        MA_max = np.nanmax(MA)
        MA_min = np.nanmin(MA)
        MA = (MA - MA_min) / (MA_max - MA_min)

        # 1
        MAD = sma(sma_MAD)
        MAD = np.diff(np.hstack((0, MAD)))
        MAD_max = np.nanmax(MAD)
        MAD_min = np.nanmin(MAD)
        MAD = (MAD - MAD_min) / (MAD_max - MAD_min)

        '''
        #cowen RNG BS
        korcas=[]
        for i in price2.index.tolist():
            delta = price2.index[i]
            cas = 1/(math.sqrt(delta+1))
            korcas.append(cas)
    
        cow = ((4*math.pi)**2) * sma(sma_cow) * korcas
        #cow = np.log(cow)
        cow_max = np.nanmax(cow)
        cow_min = np.nanmin(cow)
        cow = (cow - cow_min) / (cow_max - cow_min)
        #print (cow)
        '''


        #2
        vol = bn.move_mean(volume2, window=volvar)
        vol0_max = np.nanmax(vol)
        vol0_min = np.nanmin(vol)
        vol = (vol - vol0_min) / (vol0_max - vol0_min)


        # 2.5
        '''
        force = vol = (bn.move_mean(volume2, window=volvar)) * price2
        force = np.log(force)
        force_max = np.nanmax(force)
        force_min = np.nanmin(force)
        force = (force - force_min) / (force_max - force_min)
        '''

        # 3
        n = RSI
        delta = np.diff(np.hstack((0, price2)))
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        RolUp = bn.move_mean(dUp, window=n)
        RolDown = np.absolute(bn.move_mean(dDown, window=n))
        RS = RolUp / RolDown
        RS = np.log(RS)
        RS_max = np.nanmax(RS)
        RS_min = np.nanmin(RS)
        RS = (RS - RS_min) / (RS_max - RS_min)

        #4
        def func(x, p1, p2):
            return p1 * np.log(x) + p2

        data1 = price2
        data1 = data1[data1 > 0]
        xdata = np.array([x + 1 for x in range(len(data1))])
        ydata = np.log(data1)

        popt, pcov = curve_fit(func, xdata, ydata, p0=(3.0, -10))
        log_reg = func(xdata, popt[0], popt[1])
        log_reg0 = np.exp(log_reg)
        log_reg0 = np.log(price2) / log_reg0
        log_reg0_max = np.nanmax(log_reg0)
        log_reg0_min = np.nanmin(log_reg0)
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

        '''
        # 5
        risk55 = sma(sma_risk55)
        ri = price2 / risk55
        risk55 = np.diff(np.hstack((0, ri)))

        risk55_max = np.nanmax(risk55)
        risk55_min = np.nanmin(risk55)
        risk55 = (risk55 - risk55_min) / (risk55_max - risk55_min)
        '''
        #SUM
        risk = (MA + MAD + RS + vol)
        risk_max = np.nanmax(risk)
        risk_min = np.nanmin(risk)
        risk = (risk - risk_min) / (risk_max - risk_min)

        '''
        print("MA", MA)
        print("MAD", MAD)
        #print("force", force)
        print("RSI", RS)
        print("risk55", risk55)
        print("log_reg0", log_reg0)
        #print("risk:", risk)
        '''

    def risk_opti(risk):
        global stBTClist
        global fondlist
        global fond0
        global stBTC
        global cena
        global vlozek0

        if np.isnan(risk).all():
            pass
        else:
            risk0 = risk[-1]
            cena = price2[-1]
            if math.isnan(risk0):
                pass
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

    glassnode_price()
    glassnode_VOL()

    maxi = []
    indie = []
    index_of_params = []
    stBTClist = []
    fondlist = []
    zap=0

    sma_mala = 0
    sma_velika = 0
    sma_MAD = 0
    volvar = 0
    RSI = 0
    sma_risk55 = 0

    parametri = [sma_mala, sma_velika, sma_MAD, volvar, RSI]
    sprem = [20, 50, 70, 140, 200]
    value = [p for p in itertools.product(sprem, repeat=len(parametri))]
    print("iteracije:", len(value))

    def risk_get(value):
        global zap
        global vlozek0
        global fond0
        global stBTC
        global na
        global cena
        for risk_var in value:
            risk_var=list(risk_var)
            fond0 = 0
            stBTC = 0
            cena = 0
            cikel1 = datet.now()
            if risk_var[0] == risk_var[1]:
                pass
            elif risk_var[0] > risk_var[1]:
                pass
            else:
                global timeprice
                global timevol
                for i in timeprice[::na]:
                    global price2
                    global volume2
                    price2 = price[(timeprice < i)]
                    volume2 = volume[(timevol < i)]
                    if len(price2) < max(sprem):
                        pass
                    else:
                        calculate_risk(*risk_var)
                        if np.isnan(risk).all():
                            pass
                        else:
                            risk_opti(risk)

            stBTClist.append(stBTC * cena)
            fondlist.append(fond0)
            index_of_params.append(risk_var)
            global zipped
            zipped = [x + y for x, y in zip(stBTClist, fondlist)]

            maximual = np.amax(zipped)

            maxi.append(maximual)

            indeks = zipped.index(np.amax(zipped))

            indie.append(indeks)
            '''
            print("maximual:",maximual)
            print("index:", indeks)
            print("parametri:", index_of_params[max(indie)])
            print("obdelana:", risk_var)
            zap+=1
            print("zaporedna:", zap)
            print("cikel1:", datet.now() - cikel1)
            print("---------------------------------------------------------")
            '''
    risk_get(value)

    global zipped
    stBTClist=stBTClist[np.amax(indie)]
    fondlist=fondlist[np.amax(indie)]
    zipped=zipped[np.amax(indie)]

    print("DYNAMIC DCA")
    print("Obdobje:", "od", start_date, "do", end_date)
    print("Value of inv.:", round(stBTClist, 2))
    print("Cost of inv.:", round(fondlist,2))
    print("Delta of inv.:", round(zipped, 2))
    ROI=round((stBTClist-abs(fondlist))/abs(fondlist),2)
    print("ROI:",ROI)
    print("rezultati:", index_of_params[np.amax(indie)])
    print("parametri:", sprem)
    print("cas izracuna", datet.now() - startTime)
    print("")
    import return_on_straight_DCA_ris_learn
    return_on_straight_DCA_ris_learn


else:
    pass