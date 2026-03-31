% sqr_rnd_trial
; REGENHU
; http://www.regenhu.com/
; Translator 1.0.0.0
; Date 2026-03-30
; ESTIMATED_PRINTING_TIME 0:00:44.7962463

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
G805[-27.120, 28.955, 2.870] ; Set G55 origin
G55

; Changing tool to 'PSD 1'
#FLUSH WAIT
T1
G807[1, 0.002, 0.002] ; Enable time-based start (0.002) / stop (0.002) delays [s]
M200=1200 ; Set pressure to 120kPa
F10.000
M302 ; Wait for tool temperature
G00 G55 X-3.000 Y-3.000
M151 ; Engage tool for printing
Z0.600
M110=180 ; Set printing progress to 18%
M160 ; Turn on dispensing
G01 X3.000
M110=200 ; Set printing progress to 20%
Y3.000
M110=210 ; Set printing progress to 21%
X-3.000
M110=230 ; Set printing progress to 23%
Y-3.000
M110=240 ; Set printing progress to 24%
M161 ; Turn off dispensing
G805[-7.820, 28.955, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=250 ; Set printing progress to 25%
X-3.000
M110=270 ; Set printing progress to 27%
Z0.600
M110=280 ; Set printing progress to 28%
M160 ; Turn on dispensing
G01 X3.000
M110=290 ; Set printing progress to 29%
Y3.000
M110=310 ; Set printing progress to 31%
X-3.000
M110=320 ; Set printing progress to 32%
Y-3.000
M110=330 ; Set printing progress to 33%
M161 ; Turn off dispensing
G805[-7.820, -9.645, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=350 ; Set printing progress to 35%
Y-3.000
M110=370 ; Set printing progress to 37%
Z0.600
M110=380 ; Set printing progress to 38%
M160 ; Turn on dispensing
G01 X3.000
M110=390 ; Set printing progress to 39%
Y3.000
M110=410 ; Set printing progress to 41%
X-3.000
M110=420 ; Set printing progress to 42%
Y-3.000
M110=430 ; Set printing progress to 43%
M161 ; Turn off dispensing
G805[11.480, -9.645, 2.870] ; Set G55 origin
G55
G00 Z18.400
M110=450 ; Set printing progress to 45%
X-3.000
M110=460 ; Set printing progress to 46%
Z0.600
M110=470 ; Set printing progress to 47%
M160 ; Turn on dispensing
G01 X3.000
M110=490 ; Set printing progress to 49%
Y3.000
M110=500 ; Set printing progress to 50%
X-3.000
M110=510 ; Set printing progress to 51%
Y-3.000
M110=530 ; Set printing progress to 53%
M161 ; Turn off dispensing
G805[-27.120, 28.955, 2.870] ; Set G55 origin
G55

; Changing tool to 'Curing'
#FLUSH WAIT
G803 ; Move to safe height
T6
G807 ; Disable start/stop delays
F21.651
G00 G55 X0.000 Y0.000
M151 ; Engage tool for printing
Z3.070
M162=305 ; Dispense for 305ms
M110=540 ; Set printing progress to 54%
M110=620 ; Set printing progress to 62%
G805[-7.820, 28.955, 2.870] ; Set G55 origin
G55

; Changing tool to 'Curing'
#FLUSH WAIT
G803 ; Move to safe height
T6
G807 ; Disable start/stop delays
F21.651
G00 G55 X0.000 Y0.000
M151 ; Engage tool for printing
Z3.070
M162=305 ; Dispense for 305ms
M110=630 ; Set printing progress to 63%
M110=710 ; Set printing progress to 71%
G805[-7.820, -9.645, 2.870] ; Set G55 origin
G55

; Changing tool to 'Curing'
#FLUSH WAIT
G803 ; Move to safe height
T6
G807 ; Disable start/stop delays
F21.651
G00 G55 X0.000 Y0.000
M151 ; Engage tool for printing
Z3.070
M162=305 ; Dispense for 305ms
M110=720 ; Set printing progress to 72%
M110=790 ; Set printing progress to 79%

; Changing tool to 'Curing'
#FLUSH WAIT
G803 ; Move to safe height
T6
G807 ; Disable start/stop delays
F21.651
G00 G55 X19.300 Y0.000
M151 ; Engage tool for printing
Z3.070
M162=305 ; Dispense for 305ms
M110=810 ; Set printing progress to 81%
M110=880 ; Set printing progress to 88%

#FLUSH WAIT
G800 ; Go home
M110=1000 ; Set printing progress to 100%
M30
