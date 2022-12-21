def risk_calc(value):
    for risk_var in value:
        risk_var=list(risk_var)
        fond0 = 0
        vlozek0 = 50
        stBTC = 0
        na = 7
        cena = 0
        cikel1 = datet.now()
        '''
        for i, v in enumerate(parametri):
            risk_var.append(razpon[i])
            print(razpon[i])
            print(risk_var)
        '''
        if risk_var[0] == risk_var[1]:
            pass
        else:
            for i in data["t"][::na]:
                data2 = data[(data['t'] < i)]
                vol2 = vol0[data["t"] < i]
                #print(data2)
                #print(len(data2))

                if len(data2) < 151:
                    pass
                    #print("deladeladeladeladeladeladeladeladeladela")
                else:
                    #print(data2)
                    calculate_risk(*risk_var)
                    #print(stBTC)
                    if risk.isnull().all():
                        pass
                    else:
                        #cikel4 = datet.now()
                        # print(*risk_var)
                        #print(risk)
                        risk_opti(risk)
                        #print(stBTC)
                        #print(stBTC)
                        #print(fond0)

        stBTClist.append(stBTC * cena)
        fondlist.append(fond0)

        index_of_params.append(risk_var)
        #print("stBTClist", stBTClist)
        #print("fondlist", fondlist)

        zipped = [x + y for x, y in zip(stBTClist, fondlist)]
        #print("ipped:", zipped)
        #print(zipped)
        maximual = max(zipped)
        print("maximual:",maximual)
        maxi.append(maximual)
        #print(maxi)

        indeks = zipped.index(max(zipped))
        #indeks2 = min([(j, i) for i, j in enumerate(maximual)])
        print("index:", indeks)
        indie.append(indeks)
        #print(indie)
        print("parametri:", index_of_params[max(indie)])
        print("obdelana:", risk_var)
        zap+=1
        print("zaporedna:", zap)
        # print("max:", maximual)
        # print("indeks:", indeks)
        print("cikel1:", datet.now() - cikel1)
