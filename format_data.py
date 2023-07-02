import numpy as np
import glob, os

no_samp = 10

def format_data(raw_data, no_samp):
    rawx = np.array([])
    rawy = np.array([])
    rawz = np.array([])
    formatData = np.array([])
    for line in raw_data:
        line = line.split(';')
        for i in range(no_samp):
            try:
                rawx = np.append(rawx, float(line[i*3 + 0]))
                rawy = np.append(rawy, float(line[i*3 + 1]))
                rawz = np.append(rawz, float(line[i*3 + 2]))
            except:
                print("Esantion incomplet")
                if len(rawx) != len(rawy) or len(rawz) != len(rawy):
                    len_data = min(len(rawx), len(rawy), len(rawz))
                    rawx = rawx[:len_data]
                    rawy = rawy[:len_data]
                    rawz = rawz[:len_data]
    formatData = np.vstack((rawx, rawy, rawz))
    return formatData

def format_npy(nume, no_samp):
    raw_data = np.load(nume)
    formatData = format_data(raw_data, no_samp)
    numpyname = nume[:-4] + "f.npy"
    with open(numpyname, 'wb') as npyfile:
        np.save(npyfile, formatData)

def cutdata(nume, nr_data):
    format_data = np.load(nume)
    datex = format_data[0]
    datey = format_data[1]
    datez = format_data[2]
    datex = datex[2 * 640:2 * 640 + nr_data]
    datey = datey[2 * 640:2 * 640 + nr_data]
    datez = datez[2 * 640:2 * 640 + nr_data]
    cutData = np.vstack((datex, datey, datez))
    numpyname = nume[:-4] + "c.npy"
    with open(numpyname, 'wb') as npyfile:
        np.save(npyfile, cutData)

def esantioaneNebune(nume):
    Data = np.load(nume)
    cutData = Data
    if cutData[0][-1] - cutData[0][-2] > 10 or cutData[0][0] - cutData[0][1] > 10:
        print(f"MAX: {cutData[0][-1]}, {cutData[0][-2]} , min: {cutData[0][0]},{cutData[0][1]}, diferenta: {cutData[0][-1] - cutData[0][-2]}, {cutData[0][0] - cutData[0][1]}")
        print("Axa X e paaa")
        print(nume)
    if cutData[1][-1] - cutData[1][-2] > 10 or cutData[1][0] - cutData[1][1] > 10:
        print(f"MAX: {cutData[1][-1]}, {cutData[1][-2]} , min: {cutData[1][0]},{cutData[1][1]}, diferenta: {cutData[1][-1] - cutData[1][-2]}, {cutData[1][0] - cutData[1][1]}")
        print("Axa Y e paaa")
        print(nume)
    if cutData[2][-1] - cutData[2][-2] > 10 or cutData[2][0] - cutData[2][1] > 10:
        print(f"MAX: {cutData[2][-1]}, {cutData[2][-2]} , min: {cutData[2][0]},{cutData[2][1]}, diferenta: {cutData[2][-1] - cutData[2][-2]}, {cutData[2][0] - cutData[2][1]}")
        print("Axa Z e paaa")
        print(nume)
    print("**********************************************************")
    print()

def winwin(nume, win, overlap, pad):
    Data = np.load(nume)
    no_samples = len(Data[0])
    shape = np.shape(Data)
    if shape == (3, 38400):
        no_samples_left = (no_samples - overlap ) % (win - overlap)
        if no_samples_left != 0:
            if pad == 0 :
                limit = no_samples - no_samples_left
                Data[0] = Data[0][:limit]
                Data[1] = Data[1][:limit]
                Data[2] = Data[2][:limit]
                no_samples = len(Data[0])
            else:
                add = np.zeros(win - overlap - no_samples_left)
                Data[0] = np.append(Data[0], add)
                Data[1] = np.append(Data[1], add)
                Data[2] = np.append(Data[2], add)
                no_samples = len(Data[0])
        for i in range(0, (no_samples - overlap), (win - overlap)):
            if i == 0:
                winn = np.hstack((Data[0][i:i+win], Data[1][i:i+win], Data[2][i:i+win]))
            else:
                row = np.hstack((Data[0][i:i+win], Data[1][i:i+win], Data[2][i:i+win]))
                winn = np.vstack((winn, row))
        print(np.shape(winn))
        numpynamex = nume[:-4] + "x.npy"
        with open(numpynamex, 'wb') as npyfile:
            np.save(npyfile, winn)

        label = int(nume[-12:-11])
        labels = np.ones(np.shape(winn)[0]) * label
        numpynamey = nume[:-4] + "y.npy"
        with open(numpynamey, 'wb') as npyfile:
            np.save(npyfile, labels)
    else:
        print(f"{nume} are forma {np.shape(Data)}")


def isNaN(num):
    if float('-inf') < float(num) < float('inf'):
        return num
    else:
        return 0

def MAV (signal):
    result = 0
    for i in range(signal.size):
        if signal[i] < 0:
            result = result - signal[i]
        else:
            result = result + signal[i]
    return result / signal.size

def ZeroCrossingRate(signal, alfa):
    result = 0
    for i in range (1, len(signal)):
        if ( signal[i] * signal [i-1] < 0 ) and ( abs(signal[i] - signal[i-1]) > alfa ):
                result = result + 1
    return result

def Skewness(signal):
    result = 0
    for i in range(len(signal)):
        result = result + ((signal[i] - np.mean(signal)) / (np.std(signal) + np.spacing(1))) ** 3
        result = isNaN(result)
    result = result / len(signal)
    return result

def WaveformLength(signal):
    result = 0
    for i in range(1, len(signal)):
        result = result + abs(signal[i] - signal[i-1])
    return result

def StandardDeviation(signal):
    media = 0
    for i in range(len(signal)):
        media = media + signal[i];
    media = media / len(signal)
    result = 0
    for i in range(len(signal)):
        result = result + (signal[i] - media)**2
    result = result / len(signal)
    result = np.sqrt(result)
    return result

def caracteristici(nume):
    win = int(0.5 * 640)
    alfa = 0.2
    date = np.load(nume)
    for ind, line in enumerate(date):
        for i in range(3):
            if i == 0:
                row_cell = np.array([MAV(line[i*win:(i+1)*win]), ZeroCrossingRate(line[i*win:(i+1)*win], alfa), Skewness(line[i*win:(i+1)*win]), WaveformLength(line[i*win:(i+1)*win]),StandardDeviation(line[i*win:(i+1)*win])])
            else:
                ch_cell = np.array([MAV(line[i*win:(i+1)*win]), ZeroCrossingRate(line[i*win:(i+1)*win], alfa), Skewness(line[i*win:(i+1)*win]), WaveformLength(line[i*win:(i+1)*win]), StandardDeviation(line[i*win:(i+1)*win])])
                row_cell = np.append(row_cell, ch_cell)
        if ind == 0:
            dateTrs = row_cell
        else:
            dateTrs = np.vstack((dateTrs, row_cell))
    numpynamex = nume[:-4] + "X.npy"
    with open(numpynamex, 'wb') as npyfile:
        np.save(npyfile, dateTrs)

def getuser(nume):
    return int(nume[3:5])

def getclass(nume):
    return int(nume[1:2])

def alldataX(allnumeX):
    ordernumeX = allnumeX.copy()
    for Xnpy in allnumeX:
        indx = 7*(getuser(Xnpy[-15:])-1) + getclass(Xnpy[-15:])
        ordernumeX[indx] = Xnpy[-15:]
    return ordernumeX

def alldatay(allnumey):
    ordernumey = allnumey.copy()
    for ynpy in allnumey:
        indx = 7*(getuser(ynpy[-14:])-1) + getclass(ynpy[-14:])
        ordernumey[indx] = ynpy[-14:]
    return ordernumey

param = np.array([])
def normal(signal):
    global param
    print(f"Media: {np.mean(signal)}, iar dispersia {np.std(signal)}")
    param = np.hstack((param, np.array([np.mean(signal), np.std(signal)]).T))
    return (signal - np.mean(signal))/(np.std(signal) + np.spacing(1))

def normalize(data):
    for i in range(np.shape(data)[1]):
        if i == 0:
            col = normal(data[:, i])
        else:
            col = np.vstack((col, normal(data[:, i])))
    return col.T

def savenpy(name, date):
    with open(name, 'wb') as npyfile:
        np.save(npyfile, date)

def userdata(Xname, Yname):
    Xdata = np.load(Xname)
    Ydata= np.load(Yname)

    Xtrain = Xdata[:15*239 * 7, :]
    print(np.shape(Xtrain))
    np.save("XU_train.npy", Xtrain)

    Ytrain = Ydata[:15 * 239 * 7]
    print(np.shape(Ytrain))
    np.save("YU_train.npy", Ytrain)

    Xval = Xdata[15*239 * 7:(15+4)*239 * 7, :]
    print(np.shape(Xval))
    np.save("XU_val.npy", Xval)

    Yval = Ydata[15*239 * 7:(15+4)*239 * 7]
    print(np.shape(Yval))
    np.save("YU_val.npy", Yval)

    Xtest = Xdata[(15 + 4) * 239 * 7:, :]
    print(np.shape(Xtest))
    np.save("XU_test.npy", Xtest)

    Ytest = Ydata[(15 + 4) * 239 * 7:]
    print(np.shape(Ytest))
    np.save("YU_test.npy", Ytest)
    
def intrauserdata(Xname, Yname):
    
    Xdata = np.load(Xname)
    Ydata= np.load(Yname)
    
    for i in range(0, len(Ydata), 239):
        #167 train; 48 val; 24 test
        if i == 0:
            Xtrain = Xdata[i:i+167, :]
            Ytrain = Ydata[i:i+167]
            
            Xval = Xdata[i+167:i+167+48, :]
            Yval = Ydata[i+167:i+167+48]
            
            Xtest = Xdata[i+167+48:i+239, :]
            Ytest = Ydata[i+167+48:i+239]
        else:
            Xtrain = np.vstack((Xtrain, Xdata[i:i+167, :]))
            Ytrain = np.hstack((Ytrain,Ydata[i:i+167]))
            
            Xval = np.vstack((Xval, Xdata[i+167:i+167+48, :]))
            Yval = np.hstack((Yval, Ydata[i+167:i+167+48]))
            
            Xtest = np.vstack((Xtest,Xdata[i+167+48:i+239, :]))
            Ytest = np.hstack((Ytest,Ydata[i+167+48:i+239]))
            
    print(np.shape(Xtrain))
    np.save("XI_train.npy", Xtrain)

    print(np.shape(Ytrain))
    np.save("YI_train.npy", Ytrain)

    print(np.shape(Xval))
    np.save("XI_val.npy", Xval)

    print(np.shape(Yval))
    np.save("YI_val.npy", Yval)

    print(np.shape(Xtest))
    np.save("XI_test.npy", Xtest)

    print(np.shape(Ytest))
    np.save("YI_test.npy", Ytest)
    
"""    
intrauserdata("X_scaled.npy", "Y_label.npy")


userdata("X_scaled.npy", "Y_label.npy")

X_scaled = normalize(np.load("X_unscaled.npy"))

name = "X_scaled.npy"
with open(name, 'wb') as npyfile:
    np.save(npyfile, X_scaled)

name = "MedDisp.npy"
with open(name, 'wb') as npyfile:
    np.save(npyfile, param)

list_ynpys = glob.glob('./*y.npy')
ordernumey = alldatay(list_ynpys)
for i in range(len(ordernumey)):
    if i == 0:
        ally= np.load(ordernumey[i])
    else:
        ally = np.hstack((ally, np.load(ordernumey[i])))

print(np.shape(ally))
name = "Y_label.npy"
with open(name, 'wb') as npyfile:
    np.save(npyfile, ally)


list_Xnpys = glob.glob('./*X.npy')
ordernumeX = alldataX(list_Xnpys)
for i in range(len(ordernumeX)):
    if i == 0:
        allX = np.load(ordernumeX[i])
    else:
        allX = np.vstack((allX, np.load(ordernumeX[i])))

print(np.shape(allX))
name = "X_unscaled.npy"
with open(name, 'wb') as npyfile:
    np.save(npyfile, allX)

list_ynpys = glob.glob('./ 0_*y.npy')
print(list_ynpys)
print(len(list_ynpys))
ynpy = list_ynpys[2]
print(np.shape(np.load(ynpy[-14:])))

list_xnpys = glob.glob('./*x.npy')
for xnpy in list_xnpys:
    caracteristici(xnpy[-14:])

list_ynpys = glob.glob('./*y.npy')
print(len(list_ynpys))
for ynpy in list_ynpys:
    print(ynpy)

list_fcnpys = glob.glob('./*fc.npy')
print(len(list_fcnpys))
for fcnpy in list_fcnpys:
    win = 0.5*640
    over = 0.25*640
    winwin(fcnpy[-13:], int(win), int(over), 0)
    
list_fnpys = glob.glob('./*f.npy')
for fnpy in list_fnpys:
    print(fnpy[-12:])
    cutdata(fnpy[-12:], 640 * 60)

list_fcnpys = glob.glob('./*fc.npy')
print(len(list_fcnpys))
for fcnpy in list_fcnpys:
    dada = np.load(fcnpy)
    print(dada)
    print(np.shape(dada))
    
list_npys = glob.glob('./*.npy')
for npy in list_npys:
    format_npy(npy[-11:], 10)

list_fnpys = glob.glob('./*f.npy')
for fnpy in list_fnpys:
    print(fnpy[-12:])
    date = np.load(fnpy[-12:])
    print(np.shape(date))
"""

