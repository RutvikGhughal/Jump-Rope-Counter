from __future__ import print_function
import sys
import numpy as np
import pyaudio
import time
import librosa
import librosa.display
import tkinter as Tkinter
import time
from threading import Thread


import matplotlib.pyplot as plt

done = False
counter = 0

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK_SIZE = 1024 * 2
        self.input_device_index = 2
        self.p = None
        self.stream = None
        self.y = None
        self.frame_size = 100000
        self.skip_rope_counter = 0
        self.last_peak_value = float(0)

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, output=True, frames_per_buffer=self.CHUNK_SIZE, stream_callback=self.callback, input_device_index=self.input_device_index)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        global done
        if(done):
            return None, pyaudio.paContinue
        if type(self.y) is np.ndarray:
            self.y = np.append(self.y, np.frombuffer(in_data, dtype=np.float32))
            if(self.y.size > self.frame_size):
                self.y = self.y[-self.frame_size:]

        else:
            self.y = np.frombuffer(in_data, dtype=np.float32)
        librosa.feature.mfcc(self.y)
        return in_data, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            global done
            if(done):
                break

            if type(self.y) is np.ndarray:

                # Find peaks
                onset_env = librosa.onset.onset_strength(y=self.y, sr=self.RATE,
                                                         hop_length=512,
                                                         aggregate=np.median)

                peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)

                peak_values = np.zeros(peaks.size)
                for i in range(peaks.size):
                    peak_values[i] = onset_env[peaks[i]];

                # Print peaks list to console
                # print('Peaks detected at: ', librosa.frames_to_time(peaks, sr=self.RATE))
                # print('Peak values are: ', peak_values)
                # print('Number of peaks: ', peaks.size)

                if(peak_values.size>0):
                    curr_peak = peak_values[-1]
                    if(curr_peak != self.last_peak_value):
                        print(curr_peak)
                        if(curr_peak>1.75 and curr_peak<9):
                            self.last_peak_value = curr_peak
                            self.skip_rope_counter = self.skip_rope_counter+1

                # print('Skip Rope Count: ', self.skip_rope_counter, end='\r')
                global counter
                counter = self.skip_rope_counter
                sys.stdout.flush()

                if(peaks.size>0):
                    times = librosa.frames_to_time(np.arange(len(onset_env)),
                                                   sr=self.RATE, hop_length=512)

def startListening():
    audio = AudioHandler()
    audio.start()     # open the the stream
    audio.mainloop()  # main operations with librosa
    audio.stop()

# start function of the stopwatch
def Train(label):
	train['state']='disabled'
	start['state']='normal'
	reset['state']='disabled'

# Stop function of the stopwatch
def Start(label):
	global done
	done = False

	listening_thread = Thread(target=startListening)
	listening_thread.start()
	label['text']=str(counter)
	start['state']='disabled'
	reset['state']='normal'

	printing_thread = Thread(target=print_counter_loop)
	printing_thread.start()


def print_counter_loop():
	global done
	if(done):
		return
	time.sleep(0.1)
	global label
	label['text']=str(counter)
	print_counter_loop()

# Reset function of the stopwatch
def Reset(label):
	global counter
	global done
	counter=0
	done = True
	start['state']='normal'
	reset['state']='disabled'
	# label['text']='Counter has reset!'


root = Tkinter.Tk()
root.title("Skipping Rope Counter")


w = 450 # width for the Tk root
h = 125 # height for the Tk root

# get screen width and height
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen

# calculate x and y coordinates for the Tk root window
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)

# Fixing the window size.
root.geometry('%dx%d+%d+%d' % (w, h, x, y))

label = Tkinter.Label(root, text="Welcome!", fg="black", font="Verdana 30 bold")
label.pack()
f = Tkinter.Frame(root)
train = Tkinter.Button(f, text='Train', width=10, command=lambda:Train(label))
start = Tkinter.Button(f, text='Start',width=10,state='disabled', command=lambda:Start(label))
reset = Tkinter.Button(f, text='Reset',width=10, state='disabled', command=lambda:Reset(label))
f.pack(anchor = 'center',pady=5)
train.pack(side="left")
start.pack(side ="left")
reset.pack(side="left")
root.mainloop()
