import numpy as np
import glob
from pandas import *
from scipy.stats import skew
import matplotlib.pyplot as plt
import csv


def get_samples(no_sam: int, line: str):
    line = line.split(';')
    xarr = np.array([0])
    yarr = np.array([1])
    zarr = np.array([2])
    if len(line) >= no_sam * 3:
        for i in range(no_sam):
            try:
                xarr = np.append(xarr, float(line[i * 3 + 0]))
                yarr = np.append(yarr, float(line[i * 3 + 1]))
                zarr = np.append(zarr, float(line[i * 3 + 2]))
            except:
                print("Esantion incomplet")

    return xarr[1:], yarr[1:], zarr[1:]


def get_param(x: float, y: float, z: float, rad, sign):
    int_x = np.sqrt(y * y + z * z) / (x + np.spacing(1))
    int_y = np.sqrt(x * x + z * z) / (y + np.spacing(1))
    int_z = np.sqrt(x * x + y * y) / (z + np.spacing(1))

    angle_xacc = np.arctan(int_x)
    angle_yacc = np.arctan(int_y)
    angle_zacc = np.arctan(int_z)

    x = (x - np.sign(angle_xacc) * np.cos(angle_xacc))
    y = (y - np.sign(angle_yacc) * np.cos(angle_yacc))
    z = (z - np.sign(angle_zacc) * np.cos(angle_zacc))
    if sign == 0:
        angle_x = np.arctan2(np.sign(z) * np.sqrt(y * y + z * z), (x + np.spacing(1)))
        angle_y = np.arctan2(np.sign(x) * np.sqrt(x * x + z * z), (y + np.spacing(1)))
        angle_z = np.arctan2(np.sign(y) * np.sqrt(x * x + y * y), (z + np.spacing(1)))
        if angle_x < 0:
            angle_x += 2 * np.pi
        if angle_y < 0:
            angle_y += 2 * np.pi
        if angle_z < 0:
            angle_z += 2 * np.pi

        if rad == 0:
            angle_x = angle_x * 180 / np.pi;
            angle_y = angle_y * 180 / np.pi;
            angle_z = angle_z * 180 / np.pi
    elif sign == 1:
        angle_x = np.arctan2(np.sqrt(y * y + z * z), (x + np.spacing(1)))
        angle_y = np.arctan2(np.sqrt(x * x + z * z), (y + np.spacing(1)))
        angle_z = np.arctan2(np.sqrt(x * x + y * y), (z + np.spacing(1)))
        if rad == 0:
            angle_x = angle_x * 180 / np.pi;
            angle_y = angle_y * 180 / np.pi;
            angle_z = angle_z * 180 / np.pi

    result = np.array([angle_x, angle_y, angle_z, x, y, z])
    return result


def graf_para(rawdata, achiz_time, no_samp):
    angx = np.array([]);
    angy = np.array([]);
    angz = np.array([])
    accx = np.array([]);
    accy = np.array([]);
    accz = np.array([])
    for i in range(len(rawdata)):
        row_data = get_samples(no_samp, data[i])
        for k in range(len(row_data[0])):
            angles = get_param(float(row_data[0][k]), float(row_data[1][k]), float(row_data[2][k]), 0, 1)
            angx = np.append(angx, angles[0])
            angy = np.append(angy, angles[1])
            angz = np.append(angz, angles[2])
            accx = np.append(accx, angles[3])
            accy = np.append(accy, angles[4])
            accz = np.append(accz, angles[5])
    x = np.linspace(0, achiz_time, len(accx))
    fig_ang, ang = plt.subplots()
    ang.plot(x, angx, label='AngX');
    ang.plot(x, angy, label='AngY');
    ang.plot(x, angz, label='AngZ')
    ang.set_title('Grafic unghiuri');
    ang.legend()
    fig_acc, acc = plt.subplots()
    acc.plot(x, accx, label='AccX');
    acc.plot(x, accy, label='AccY');
    acc.plot(x, accz, label='AccZ')
    acc.set_title('Grafic acceleratie');
    acc.legend()
    plt.show()


def graf(datafile, achiz_time, no_samp, tempo):
    if isinstance(datafile, str) and datafile[-3:] == 'npy':
        data = np.load(datafile)
        rawx = np.array([]);
        rawy = np.array([]);
        rawz = np.array([])
        for i in range(len(data)):
            row_data = get_samples(no_samp, data[i])
            for k in range(len(row_data[0])):
                try:
                    rawx = np.append(rawx, row_data[0][k]);
                    rawy = np.append(rawy, row_data[1][k]);
                    rawz = np.append(rawz, row_data[2][k])
                except:
                    print("Am sarit un pachet")

    elif isinstance(datafile, str) and datafile[-3:] == 'csv':
        data = read_csv(datafile)
        rawx = np.array(data['RawX'].tolist());
        rawy = np.array(data['RawY'].tolist());
        rawz = np.array(data['RawZ'].tolist())

        data = np.array([rawx, rawy, rawz])
        return data

    else:
        data = datafile
        rawx = np.array([]);
        rawy = np.array([]);
        rawz = np.array([])
        for i in range(len(data)):
            row_data = get_samples(no_samp, data[i])
            for k in range(len(row_data[0])):
                try:
                    rawx = np.append(rawx, row_data[0][k]);
                    rawy = np.append(rawy, row_data[1][k]);
                    rawz = np.append(rawz, row_data[2][k])
                except:
                    print("Am sarit un pachet")
    x = np.linspace(0, achiz_time, len(rawx))
    plt.plot(x, rawx, label='RawX');
    plt.plot(x, rawy, label='RawY');
    plt.plot(x, rawz, label='RawZ')
    plt.title('Grafic raw data');
    plt.legend()
    plt.show(block=False)
    plt.pause(tempo)
    da = input("Este bun graficul? Da - 1/Nu - 0: ")
    if int(da) == 1 and isinstance(datafile, str):
        plt.savefig(datafile[:-3] + 'png')
        print("Graficul a fost salvat")
    elif int(da) == 1 and isinstance(datafile, str) == 0:
        numegrafic = input("Ce nume ii dai pozei? ")
        plt.savefig(numegrafic + '.png')
        print("Graficul a fost salvat")
    if plt.fignum_exists(1):
        plt.close()


def savedate(data, no_samp):
    nume = input("Introdu numele fișierului CSV sau -1 pentru a nu salva datele:")
    if int(nume) == -1:
        print("Datele nu au fost salvate")
    else:
        numpyname = nume + ".npy"
        with open(numpyname, 'wb') as npyfile:
            np.save(npyfile, data)
        filename = nume + ".csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['RawX', 'RawY', 'RawZ'])
            for i in range(len(data)):
                row_data = get_samples(no_samp, data[i])
                for k in range(len(row_data[0])):
                    try:
                        writer.writerow([row_data[0][k], row_data[1][k], row_data[2][k]])
                    except:
                        print("Am sarit un pachet")
        print(f"\nDatele au fost salvate în fișierele {filename} si in {numpyname}")



