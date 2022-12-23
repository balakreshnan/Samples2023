# NVidia Xavier NX Temp Warning using LED

## Monitor Temperature and glow LED if temp is too high

## Pre-requsities

- NVidia Xavier NX Dev Kit
- Bread Board
- LED Diode
- Cables to connect from GPIO to BreadBoard
- Using Pin 28, 29
- 28 is output pin
- 29 is ground pin

## Steps

1. Connect LED to BreadBoard
2. Connect pin 28 to a Pin in breadboard - i am using 35th row
3. Connect pin 29 to a Pin in breadboard - using 33rd row
4. LED Diode positive(+) is connected to 35th row and negative(-) is connected to 33rd row
5. Pin 1 is 3V power
6. Pin 9 is another ground pin
7. I am using Pin 1 and 9 to power a 3V DC motor
8. Now install the following packages

```
pip install RPi.GPIO
pip install jetson-stats
pip install matplotlib
```

9. Create a python file and paste the following code

```
import numpy as np
import logging
import numpy as np
import random
import numpy as np
from collections import deque
import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from jtop import jtop
import RPi.GPIO as GPIO
import time


with jtop() as jetson:
    xavier_nx = jetson.stats

    CPU_temperature = xavier_nx['Temp CPU']
    GPU_temperature = xavier_nx['Temp GPU']
    Thermal_temperature = xavier_nx['Temp thermal']
    # print('GPU Temp ' , GPU_temperature)
    
# Pin Definitions
output_pin = 29  # BCM pin 18, BOARD pin 12

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

    print("Starting demo now! Press CTRL+C to exit")
    print(' GPIO info ', GPIO.JETSON_INFO)
    print(' GPIO info ', GPIO.VERSION)
    curr_value = GPIO.HIGH
    try:
        while True:
            time.sleep(5)
            print('Input value ', GPIO.input(output_pin))
            with jtop() as jetson:
                xavier_nx = jetson.stats
                GPU_temperature = xavier_nx['Temp GPU']
                CPU_temperature = xavier_nx['Temp CPU']
                Thermal_temperature = xavier_nx['Temp thermal']
                print('GPU Temperature: ', GPU_temperature)
                if GPU_temperature > 40:
                    curr_value = GPIO.HIGH
                    print("Outputting {} to pin {}".format(curr_value, output_pin))
                    GPIO.output(output_pin, curr_value)
                    #curr_value ^= GPIO.HIGH
                else:
                    curr_value = GPIO.LOW
                    print("Outputting {} to pin {}".format(curr_value, output_pin))
                    GPIO.output(output_pin, curr_value)
                    curr_value ^= GPIO.HIGH
            # Toggle the output every second
            #print("Outputting {} to pin {}".format(curr_value, output_pin))
            #GPIO.output(output_pin, curr_value)
            #curr_value ^= GPIO.HIGH
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    main()
```

10. Run the python file

```
sudo python3 ledtemp.py
```

11. You should see the LED glowing if the temperature is above 40 degrees
12. Above code is retrieving the temperature for CPU, GPU and Thermal from jetson-stats library
13. Jetson-stats needs elevated privileges to run
14. When the temperature is above 40 degrees, we send a HIGH signal to pin 28
15. Which turns on the LED
16. if less than 40 send low signal to pin 28 to switch off the led
17. Programs checks every 5 seconds and controls based on above condition