import serial
import keyboard
import time
import numpy as np
import grafic as grf

achiz_time = 90
no_samp = 10
tempo = 5
elapsed_time = 0
j = 0
thestart = time.time()
data = np.array([])
line = []

try:
    port = serial.Serial("/dev/ttyUSB1", 921600, timeout=1)
except:
    port = serial.Serial("/dev/ttyUSB0", 921600, timeout=1)
port.flushInput()

def wait_for_space():
    while True:
        if keyboard.is_pressed('space'):
            break
        
print("Apasă tasta Space pentru a începe achizitia datelor.")
wait_for_space()
print("Cronometrul a început!")
start_time = time.time()
while elapsed_time < achiz_time:
        elapsed_time = time.time() - start_time
        if elapsed_time > 60 and elapsed_time < 65:
            print("Mai e puuuutin")
        try:
            line = port.readline().decode('utf-8').rstrip()
        except:
            print("NADA")
        data = np.append(data, line)
        
port.close()
if len(data) > 10:
    grf.savedate(data, no_samp)
    grf.graf(data, achiz_time, no_samp, tempo)
else:
    print("Ceva nu a mers bine!")
