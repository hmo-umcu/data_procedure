% 1x10 dots right 24 well plate 4kPa 0.4mm 500umMEW topright
; REGENHU
; http://www.regenhu.com/
; Translator 1.0.0.0
; Date 2026-03-09
; ESTIMATED_PRINTING_TIME 0:00:33.4570678

; INITIALIZATION
T1
M200=40 ; Set pressure to 4kPa
T0
G803 ; Move to safe height
; INITIALIZATION

M110=0 ; Set printing progress to 0%
T1
G801 ; Measure tool

#CONTOUR MODE ON [DEV PATH_DEV=0.08]

M312 ; Wait for work zone temperature
G805[10.000, 9.753, 2.500] ; Set G55 origin
G55

; Changing tool to '0.4mm 4kPa'
#FLUSH WAIT
T1
G807[1, 0.002, 0.002] ; Enable time-based start (0.002) / stop (0.002) delays [s]
M200=40 ; Set pressure to 4kPa
F10.000
M302 ; Wait for tool temperature
G00 G55 X-1.000 Y0.500
M151 ; Engage tool for printing
  Z0.500
M110=250 ; Set printing progress to 25%
M162=500 ; Dispense for 500ms
M110=260 ; Set printing progress to 26%
G00 Z17.000
M110=280 ; Set printing progress to 28%
X-0.500 Y1.000
  Z0.500
M110=300 ; Set printing progress to 30%
M162=500 ; Dispense for 500ms
M110=320 ; Set printing progress to 32%
G00 Z17.000
M110=330 ; Set printing progress to 33%
X0.000 Y0.500
M110=340 ; Set printing progress to 34%
  Z0.500
M110=350 ; Set printing progress to 35%
M162=500 ; Dispense for 500ms
M110=370 ; Set printing progress to 37%
G00 Z17.000
M110=390 ; Set printing progress to 39%
X0.500 Y1.000
  Z0.500
M110=410 ; Set printing progress to 41%
M162=500 ; Dispense for 500ms
M110=420 ; Set printing progress to 42%
G00 Z17.000
M110=440 ; Set printing progress to 44%
X1.000 Y0.500
  Z0.500
M110=460 ; Set printing progress to 46%
M162=500 ; Dispense for 500ms
M110=470 ; Set printing progress to 47%
G00 Z17.000
M110=490 ; Set printing progress to 49%
X0.500 Y-0.000
M110=500 ; Set printing progress to 50%
  Z0.500
M110=510 ; Set printing progress to 51%
M162=500 ; Dispense for 500ms
M110=530 ; Set printing progress to 53%
G00 Z17.000
M110=540 ; Set printing progress to 54%
X0.000 Y-0.500
M110=550 ; Set printing progress to 55%
  Z0.500
M110=570 ; Set printing progress to 57%
M162=500 ; Dispense for 500ms
M110=580 ; Set printing progress to 58%
G00 Z17.000
M110=600 ; Set printing progress to 60%
X-0.500 Y-1.000
  Z0.500
M110=620 ; Set printing progress to 62%
M162=500 ; Dispense for 500ms
M110=630 ; Set printing progress to 63%
G00 Z17.000
M110=650 ; Set printing progress to 65%
X-1.000 Y-0.500
  Z0.500
M110=670 ; Set printing progress to 67%
M162=500 ; Dispense for 500ms
M110=690 ; Set printing progress to 69%
G00 Z17.000
M110=700 ; Set printing progress to 70%
X-0.500 Y-0.000
M110=710 ; Set printing progress to 71%
  Z0.500
M110=720 ; Set printing progress to 72%
M162=500 ; Dispense for 500ms
M110=740 ; Set printing progress to 74%
G00 Z17.000
M110=760 ; Set printing progress to 76%
X0.500 Y-1.000
  Z0.500
M110=780 ; Set printing progress to 78%
M162=500 ; Dispense for 500ms
M110=790 ; Set printing progress to 79%
G00 Z17.000
M110=810 ; Set printing progress to 81%
X1.000 Y-0.500
  Z0.500
M110=830 ; Set printing progress to 83%
M162=500 ; Dispense for 500ms
M110=850 ; Set printing progress to 85%

#FLUSH WAIT
G800 ; Go home
M110=1000 ; Set printing progress to 100%
M30
