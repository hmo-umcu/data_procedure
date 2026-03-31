% data_collection_24well_4rows
; REGENHU
; http://www.regenhu.com/
; Translator 1.0.0.0
; Date 2026-03-31
; ESTIMATED_PRINTING_TIME 0:01:51.4461033

; INITIALIZATION
T1
M200=1200 ; Set pressure to 120kPa
T0
G803 ; Move to safe height
; INITIALIZATION

M110=0 ; Set printing progress to 0%
T1
G801 ; Measure tool

#CONTOUR MODE ON [DEV PATH_DEV=0.08]

M312 ; Wait for work zone temperature
G805[-46.420, 28.955, 2.870] ; Set G55 origin
G55

; Changing tool to 'PSD 1'
#FLUSH WAIT
T1
G807[1, 0.002, 0.002] ; Enable time-based start (0.002) / stop (0.002) delays [s]
M200=1200 ; Set pressure to 120kPa
F10.000
M302 ; Wait for tool temperature
G00 G55 X-2.900 Y2.900
M151 ; Engage tool for printing
Z0.600
M110=70 ; Set printing progress to 7%
M160 ; Turn on dispensing
G01 Y-2.900
M110=80 ; Set printing progress to 8%
X2.900
Y2.900
M110=90 ; Set printing progress to 9%
X-2.900
M161 ; Turn off dispensing
G805[-27.120, 28.955, 2.870] ; Set G55 origin
G55
G00 Z18.400
X-2.900
M110=100 ; Set printing progress to 10%
Z0.600
M160 ; Turn on dispensing
G01 Y-2.900
M110=110 ; Set printing progress to 11%
X2.900
Y2.900
M110=120 ; Set printing progress to 12%
X-2.900
M161 ; Turn off dispensing
G805[-7.820, 28.955, 2.870] ; Set G55 origin
G55
G00 Z18.400
X-2.900
M110=130 ; Set printing progress to 13%
Z0.600
M110=140 ; Set printing progress to 14%
M160 ; Turn on dispensing
G01 Y-2.900
X2.900
M110=150 ; Set printing progress to 15%
Y2.900
X-2.900
M161 ; Turn off dispensing
G805[11.480, 28.955, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=160 ; Set printing progress to 16%
X-2.900
Z0.600
M110=170 ; Set printing progress to 17%
M160 ; Turn on dispensing
G01 Y-2.900
X2.900
M110=180 ; Set printing progress to 18%
Y2.900
X-2.900
M161 ; Turn off dispensing
G805[30.780, 28.955, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=190 ; Set printing progress to 19%
X-2.900
Z0.600
M110=200 ; Set printing progress to 20%
M160 ; Turn on dispensing
G01 Y-2.900
X2.900
M110=210 ; Set printing progress to 21%
Y2.900
M110=220 ; Set printing progress to 22%
X-2.900
M161 ; Turn off dispensing
G805[50.080, 28.955, 2.870] ; Set G55 origin
G55
G00 Z18.400
X-2.900
M110=230 ; Set printing progress to 23%
Z0.600
M160 ; Turn on dispensing
G01 Y-2.900
M110=240 ; Set printing progress to 24%
X2.900
Y2.900
M110=250 ; Set printing progress to 25%
X-2.900
M161 ; Turn off dispensing
G805[50.080, 9.655, 2.870] ; Set G55 origin
G55
G00 Z18.400
Y2.900
M110=260 ; Set printing progress to 26%
Z0.600
M160 ; Turn on dispensing
G01 Y-2.900
M110=270 ; Set printing progress to 27%
X2.900
Y2.900
M110=280 ; Set printing progress to 28%
X-2.900
M161 ; Turn off dispensing
G805[30.780, 9.655, 2.870] ; Set G55 origin
G55
G00 Z18.400
X2.900
M110=290 ; Set printing progress to 29%
Z0.600
M160 ; Turn on dispensing
G01 X-2.900
M110=300 ; Set printing progress to 30%
Y-2.900
M110=310 ; Set printing progress to 31%
X2.900
Y2.900
M110=320 ; Set printing progress to 32%
M161 ; Turn off dispensing
G805[11.480, 9.655, 2.870] ; Set G55 origin
G55
G00 Z18.400
X2.900
M110=330 ; Set printing progress to 33%
Z0.600
M160 ; Turn on dispensing
G01 X-2.900
M110=340 ; Set printing progress to 34%
Y-2.900
X2.900
M110=350 ; Set printing progress to 35%
Y2.900
M161 ; Turn off dispensing
G805[-7.820, 9.655, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=360 ; Set printing progress to 36%
X2.900
Z0.600
M110=370 ; Set printing progress to 37%
M160 ; Turn on dispensing
G01 X-2.900
Y-2.900
M110=380 ; Set printing progress to 38%
X2.900
M110=390 ; Set printing progress to 39%
Y2.900
M161 ; Turn off dispensing
G805[-27.120, 9.655, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=400 ; Set printing progress to 40%
X2.900
Z0.600
M110=410 ; Set printing progress to 41%
M160 ; Turn on dispensing
G01 X-2.900
Y-2.900
M110=420 ; Set printing progress to 42%
X2.900
Y2.900
M110=430 ; Set printing progress to 43%
M161 ; Turn off dispensing
G805[-46.420, 9.655, 2.870] ; Set G55 origin
G55
G00 Z18.400
X2.900
M110=440 ; Set printing progress to 44%
Z0.600
M160 ; Turn on dispensing
G01 X-2.900
M110=450 ; Set printing progress to 45%
Y-2.900
X2.900
M110=460 ; Set printing progress to 46%
Y2.900
M161 ; Turn off dispensing
G805[-46.420, -9.645, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=470 ; Set printing progress to 47%
Y2.900
M110=480 ; Set printing progress to 48%
Z0.600
M160 ; Turn on dispensing
G01 X-2.900
M110=490 ; Set printing progress to 49%
Y-2.900
X2.900
M110=500 ; Set printing progress to 50%
Y2.900
M161 ; Turn off dispensing
G805[-27.120, -9.645, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=510 ; Set printing progress to 51%
X-2.900
Z0.600
M110=520 ; Set printing progress to 52%
M160 ; Turn on dispensing
G01 Y-2.900
X2.900
M110=530 ; Set printing progress to 53%
Y2.900
X-2.900
M161 ; Turn off dispensing
G805[-7.820, -9.645, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=540 ; Set printing progress to 54%
X-2.900
Z0.600
M110=550 ; Set printing progress to 55%
M160 ; Turn on dispensing
G01 Y-2.900
M110=560 ; Set printing progress to 56%
X2.900
Y2.900
M110=570 ; Set printing progress to 57%
X-2.900
M161 ; Turn off dispensing
G805[11.480, -9.645, 2.870] ; Set G55 origin
G55
G00 Z18.400
X-2.900
M110=580 ; Set printing progress to 58%
Z0.600
M160 ; Turn on dispensing
G01 Y-2.900
M110=590 ; Set printing progress to 59%
X2.900
Y2.900
M110=600 ; Set printing progress to 60%
X-2.900
M161 ; Turn off dispensing
G805[30.780, -9.645, 2.870] ; Set G55 origin
G55
G00 Z18.400
X-2.900
M110=610 ; Set printing progress to 61%
Z0.600
M160 ; Turn on dispensing
G01 Y-2.900
M110=620 ; Set printing progress to 62%
X2.900
Y2.900
M110=630 ; Set printing progress to 63%
X-2.900
M161 ; Turn off dispensing
G805[50.080, -9.645, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=640 ; Set printing progress to 64%
X-2.900
Z0.600
M110=650 ; Set printing progress to 65%
M160 ; Turn on dispensing
G01 Y-2.900
X2.900
M110=660 ; Set printing progress to 66%
Y2.900
X-2.900
M161 ; Turn off dispensing
G805[50.080, -28.945, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=670 ; Set printing progress to 67%
X2.900 Y2.900
Z0.600
M110=680 ; Set printing progress to 68%
M160 ; Turn on dispensing
G01 X-2.900
Y-2.900
M110=690 ; Set printing progress to 69%
X2.900
Y2.900
M110=700 ; Set printing progress to 70%
M161 ; Turn off dispensing
G805[30.780, -28.945, 2.870] ; Set G55 origin
G55
G00 Z18.400
X2.900
M110=710 ; Set printing progress to 71%
Z0.600
M110=720 ; Set printing progress to 72%
M160 ; Turn on dispensing
G01 X-2.900
Y-2.900
M110=730 ; Set printing progress to 73%
X2.900
Y2.900
M110=740 ; Set printing progress to 74%
M161 ; Turn off dispensing
G805[11.480, -28.945, 2.870] ; Set G55 origin
G55
G00 Z18.400
X2.900
M110=750 ; Set printing progress to 75%
Z0.600
M160 ; Turn on dispensing
G01 X-2.900
M110=760 ; Set printing progress to 76%
Y-2.900
X2.900
M110=770 ; Set printing progress to 77%
Y2.900
M161 ; Turn off dispensing
G805[-7.820, -28.945, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=780 ; Set printing progress to 78%
X2.900
Z0.600
M110=790 ; Set printing progress to 79%
M160 ; Turn on dispensing
G01 X-2.900
M110=800 ; Set printing progress to 80%
Y-2.900
X2.900
M110=810 ; Set printing progress to 81%
Y2.900
M161 ; Turn off dispensing
G805[-27.120, -28.945, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=820 ; Set printing progress to 82%
X2.900
Z0.600
M110=830 ; Set printing progress to 83%
M160 ; Turn on dispensing
G01 X-2.900
Y-2.900
M110=840 ; Set printing progress to 84%
X2.900
Y2.900
M110=850 ; Set printing progress to 85%
M161 ; Turn off dispensing
G805[-46.420, -28.945, 2.870] ; Set G55 origin
G55
G00 Z18.400
X2.900
M110=860 ; Set printing progress to 86%
Z0.600
M160 ; Turn on dispensing
G01 X-2.900
M110=870 ; Set printing progress to 87%
Y-2.900
M110=880 ; Set printing progress to 88%
X2.900
Y2.900
M110=890 ; Set printing progress to 89%
M161 ; Turn off dispensing

#FLUSH WAIT
G800 ; Go home
M110=1000 ; Set printing progress to 100%
M30
