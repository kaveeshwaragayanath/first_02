#!/usr/bin/env python3
""" Transmits a tone out of the AIR-T and simultaneously receives data with AGC enabled, 
    then saves the final received spectrum plot. """
import sys
import numpy as np
import argparse
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_RX, SOAPY_SDR_CS16, SOAPY_SDR_OVERFLOW, errToStr
import concurrent.futures
import threading
import time
import matplotlib.pylab as plt # Re-enabled plotting

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

def calculate_tone_snr(s_cplx):
    """ Calculates a simplified peak SNR from the complex data using FFT. """
    N = s_cplx.size
    S_fft = np.fft.fft(s_cplx)
    S_abs_sq = np.abs(S_fft)**2
    
    # Find the peak tone magnitude (Signal Power)
    peak_idx = np.argmax(S_abs_sq)
    peak_power = S_abs_sq[peak_idx]
    
    # Estimate Noise Power by removing the peak and calculating the average
    noise_indices = np.arange(N)
    noise_indices = np.delete(noise_indices, [peak_idx, 0]) 
    
    if noise_indices.size > 0:
        noise_power = np.mean(S_abs_sq[noise_indices])
        snr_linear = peak_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
    else:
        snr_db = -999.0
        
    return snr_db

def plot_spectrum(s_cplx, freq, fs):
    """ Performs FFT and saves the received spectrum plot. """
    N = s_cplx.size
    
    # 1. Compute FFT and shift zero frequency to center
    S_fft = np.fft.fftshift(np.fft.fft(s_cplx))
    
    # 2. Calculate Frequencies
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    
    # 3. Calculate Power Spectral Density (in dBFS)
    Pxx_dbfs = 20 * np.log10(np.abs(S_fft) / N + 1e-12) 

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e6, Pxx_dbfs) # Plot frequency in MHz
    
    plt.axvline(x=(freq - freq) / 1e6, color='r', linestyle='--', label=f'Tone Center (0 MHz Baseband)')
    plt.xlim(-fs/2/1e6, fs/2/1e6)
    plt.ylim(np.max(Pxx_dbfs) - 80, np.max(Pxx_dbfs) + 5)
    plt.title(f'Received Spectrum (LO: {freq/1e6:.2f} MHz, Fs: {fs/1e6:.2f} MSPS)')
    plt.xlabel('Baseband Frequency (MHz)')
    plt.ylabel('Amplitude (dBFS)')
    plt.legend()
    plt.grid(True)
    
    # --- SAVE THE PLOT TO FILE ---
    output_filename = 'final_spectrum.png'
    plt.savefig(output_filename)
    print(f"\n 💾 Spectrum saved to {output_filename}")
    plt.close() # Close the figure to free memory

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
    
    # **ENABLE AGC for RX**
    sdr.setGainMode(SOAPY_SDR_RX, chan, True)
    print("RX Activated: AGC Enabled")

    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [chan])
    sdr.activateStream(rx_stream)
    
    # Create RX buffer (NumPy array for interleaved int16 samples)
    rx_buff = np.empty(2 * buff_len, dtype=np.int16)
    
    # Variable to hold the last successfully captured complex buffer
    last_s_cplx = np.array([]) 

    # 3. Launch TX Thread
    tx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    tx_stop_event = threading.Event()
    tx_task = tx_executor.submit(tx_task_fn, sdr, tx_stream, tx_buff, buff_len, tx_stop_event)
    tx_task.add_done_callback(lambda task: sdr.deactivateStream(tx_stream))

    # 4. Main Thread: Continuous RX and Processing
    print('\nStarting simultaneous TX/RX. Press Ctrl+C to stop.')
    print('RX Status | Samples Read | Power (dBFS) | Tone SNR (dB)')
    
    try:
        while not tx_stop_event.is_set():
            # Read data from the receiver stream
            sr = sdr.readStream(rx_stream, [rx_buff], buff_len, timeoutUs=100000)
            
            if sr.ret < 0:
                if sr.ret == SOAPY_SDR_OVERFLOW:
                    status_str = 'O'
                else:
                    status_str = 'E' 
                print(f'RX Status: {status_str}', end='\r')
                continue
            
            # --- RX DATA PROCESSING ---
            s_int16 = rx_buff[:2 * sr.ret]
            s_cplx = s_int16[0::2].astype(np.float32) + 1j * s_int16[1::2].astype(np.float32)
            s_cplx = s_cplx / 32767.0 # Normalize to 1.0
            
            # Save the last valid complex buffer for plotting on shutdown
            last_s_cplx = s_cplx 

            # Calculate mean power
            mean_power = np.mean(np.abs(s_cplx) ** 2)
            power_dbfs = 10 * np.log10(mean_power + 1e-12) 
            
            # Check for tone presence (SNR)
            tone_snr = calculate_tone_snr(s_cplx)

            # Display status
            print(f'OK        | {sr.ret:12} | {power_dbfs:.2f} dBFS | {tone_snr:.2f} dB', end='\r')
            
    except KeyboardInterrupt:
        # 5. Shutdown and Plotting Triggered by Ctrl+C
        print("\nStopping streams...")
        
    except Exception as e:
        print(f'\nAn error occurred in RX loop: {e}')
        
    finally:
        # Ensure TX thread stops regardless of exception
        tx_stop_event.set()
        try:
            tx_task.result(timeout=1.0)
        except concurrent.futures.TimeoutError:
            pass
        
        sdr.deactivateStream(rx_stream)
        sdr.closeStream(tx_stream)
        sdr.closeStream(rx_stream)
        
        print('\nFull-duplex streaming stopped. Streams closed.')
        
        # --- FINAL PLOT OF THE LAST RECEIVED BUFFER ---
        if last_s_cplx.size > 0:
            plot_spectrum(last_s_cplx, freq, fs)
        else:
            print("No data received for plotting.")

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
                        default=30.0, help='RX gain (Argument value ignored, AGC is forced)')
    parser.add_argument('-n', type=int, required=False, dest='buff_len',
                        default=16384, help='Buffer Size')
    return parser.parse_args(sys.argv[1:])

if __name__ == '__main__':
    pars = parse_command_line_arguments()
    tx_rx_tone(pars.freq, pars.chan, pars.fs, pars.gain, pars.rx_gain, pars.buff_len)