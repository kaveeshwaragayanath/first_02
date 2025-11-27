#!/usr/bin/env python3
""" Transmits a tone out of the AIR-T and simultaneously receives data. """
import sys
import numpy as np
import argparse
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_RX, SOAPY_SDR_CS16, SOAPY_SDR_OVERFLOW, errToStr
import concurrent.futures
import threading
import time

def make_tone(n, fcen, fs, phi=0.285):
    """ Generates tone signal window with a frequency that is an integer
    multiple of the sample rate so it can be repeated without a phase
    discontinuity. """
    period = fs / fcen
    assert n % period == 0, 'Total samples not integer number of periods'
    # Make Complex Valued Tone Signal (Normalized amplitude)
    wt = np.array(2 * np.pi * fcen * np.arange(n) / fs)
    sig_cplx = np.exp(1j * (wt + phi))
    # Convert to interleaved signed short integer (CS16) values
    sig_int16 = np.empty(2 * n, dtype=np.int16)
    # Scale to full 16-bit range (max 32767)
    sig_int16[0::2] = 32767 * sig_cplx.real
    sig_int16[1::2] = 32767 * sig_cplx.imag
    return sig_int16

def tx_task_fn(sdr, tx_stream, tx_buff, buff_len, stop_tx_event):
    """ Sends same buffer to transmitter indefinitely in while loop """
    while not stop_tx_event.is_set():
        # tx_buff is the interleaved int16 NumPy array
        rc = sdr.writeStream(tx_stream, [tx_buff], buff_len)
        if rc.ret != buff_len:
            raise IOError('Tx Error {}: {}'.format(rc.ret, errToStr(rc.ret)))

def tx_rx_tone(freq, chan=0, fs=31.25e6, gain=0, rx_gain=30, buff_len=16384):
    """ Transmit a tone and simultaneously receive data """
    
    # 1. Signal Preparation
    bb_freq = fs / 8  # Baseband frequency of tone
    tx_buff = make_tone(buff_len, bb_freq, fs)
    lo_freq = freq - bb_freq  # LO frequency
    
    # 2. Setup Radio
    sdr = SoapySDR.Device()  # Create AIR-T instance

    # --- TX Setup (Using CS16) ---
    sdr.setSampleRate(SOAPY_SDR_TX, chan, fs)
    sdr.setFrequency(SOAPY_SDR_TX, chan, lo_freq)
    sdr.setGain(SOAPY_SDR_TX, chan, gain)
    tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16, [chan])
    sdr.activateStream(tx_stream) 
    print(f"TX Activated: Freq={freq/1e6:.2f} MHz, Gain={gain} dB")

    # --- RX Setup (Using CS16 for consistency) ---
    sdr.setSampleRate(SOAPY_SDR_RX, chan, fs)
    sdr.setFrequency(SOAPY_SDR_RX, chan, lo_freq) # Tune RX LO to match TX
    
    # Set RX Gain (Manual gain is used based on '-r' argument)
    sdr.setGainMode(SOAPY_SDR_RX, chan, False) # Disable AGC
    sdr.setGain(SOAPY_SDR_RX, chan, rx_gain)
    print(f"RX Activated: Freq={freq/1e6:.2f} MHz, Gain={rx_gain} dB (Manual)")
    
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [chan])
    sdr.activateStream(rx_stream)
    
    # Create RX buffer (NumPy array for interleaved int16 samples)
    rx_buff = np.empty(2 * buff_len, dtype=np.int16)

    # 3. Launch TX Thread
    tx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    tx_stop_event = threading.Event()
    tx_task = tx_executor.submit(tx_task_fn, sdr, tx_stream, tx_buff, buff_len, tx_stop_event)
    tx_task.add_done_callback(lambda task: sdr.deactivateStream(tx_stream))

    # 4. Main Thread: Continuous RX and Processing
    print('\nStarting simultaneous TX/RX. Press Ctrl+C to stop.')
    print('RX Status | Samples Read | Mean Power (dBFS)')
    
    try:
        while not tx_stop_event.is_set():
            # Read data from the receiver stream
            # Timeout is critical for non-blocking read in full-duplex setup
            sr = sdr.readStream(rx_stream, [rx_buff], buff_len, timeoutUs=100000)
            
            if sr.ret < 0:
                if sr.ret == SOAPY_SDR_OVERFLOW:
                    status_str = 'O'
                else:
                    status_str = 'E' # Error
                print(f'RX Status: {status_str}', end='\r')
                continue
            
            # --- RX DATA PROCESSING ---
            # Convert interleaved int16 samples back to normalized complex float
            s_int16 = rx_buff[:2 * sr.ret]
            s_cplx = s_int16[0::2] + 1j * s_int16[1::2]
            s_cplx = s_cplx.astype(np.float32) / 32767.0 # Normalize to 1.0

            # Calculate mean power (Magnitude squared)
            mean_power = np.mean(np.abs(s_cplx) ** 2)
            # Convert to dBFS 
            power_dbfs = 10 * np.log10(mean_power + 1e-12) 

            # Display status
            print(f'OK        | {sr.ret:12} | {power_dbfs:.2f} dBFS', end='\r')
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'\nAn error occurred in RX loop: {e}')

    # 5. Stop streaming and cleanup
    tx_stop_event.set()
    try:
        tx_task.result(timeout=1.0) # Wait for TX thread to finish
    except concurrent.futures.TimeoutError:
        print("\nWarning: TX thread timeout during shutdown.")
    
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(tx_stream)
    sdr.closeStream(rx_stream)
    print('\nFull-duplex streaming stopped. Streams closed.')

def parse_command_line_arguments():
    """ Create command line options for transmit function """
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Transmit a tone on the AIR-T',
                                     formatter_class=help_formatter)
    parser.add_argument('-f', type=float, required=False, dest='freq',
                        default=1000e6, help='TX/RX Tone Frequency')
    parser.add_argument('-c', type=int, required=False, dest='chan',
                        default=0, help='TX/RX Channel Number [0 or 1]')
    parser.add_argument('-s', type=float, required=False, dest='fs',
                        default=31.25e6, help='TX/RX Sample Rate')
    parser.add_argument('-g', type=float, required=False, dest='gain',
                        default=0, help='TX gain')
    parser.add_argument('-r', type=float, required=False, dest='rx_gain',
                        default=30.0, help='RX gain (Manual setting, e.g., 30.0)')
    parser.add_argument('-n', type=int, required=False, dest='buff_len',
                        default=16384, help='Buffer Size')
    return parser.parse_args(sys.argv[1:])

if __name__ == '__main__':
    pars = parse_command_line_arguments()
    # Execute the function with the parsed arguments
    tx_rx_tone(pars.freq, pars.chan, pars.fs, pars.gain, pars.rx_gain, pars.buff_len)