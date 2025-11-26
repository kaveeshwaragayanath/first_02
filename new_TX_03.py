from typing import Tuple
import cupy as cp
from scipy.io import loadmat
#from tensorflow import keras
from cupyx.scipy.signal import fftconvolve
import matplotlib.pylab as plt

import sys
import argparse
import concurrent.futures
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_RX
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_OVERFLOW
import cupy as cp
import cupyx.scipy.signal as signal
from numba import cuda
import numpy as np

from SoapySDR import SOAPY_SDR_TX, errToStr
import concurrent.futures
import threading
import time



# --- CORRECTED FUNCTION ---
def make_tone(tx_sig: np.ndarray) -> np.ndarray:
    """ Converts complex float signal (NumPy) to interleaved real/imag int16 array. 
        tx_sig: Complex float NumPy array.
    """
    sig_cplx = tx_sig
    n = sig_cplx.size
    sig_int16 = np.empty(2 * n, dtype=np.int16)
    
    # Scale and interleave
    # 32767 is the max value for int16, used for scaling to full range
    sig_int16[0::2] = (32767 * sig_cplx.real).astype(np.int16)
    sig_int16[1::2] = (32767 * sig_cplx.imag).astype(np.int16)
    return sig_int16

def parse_command_line_arguments():
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Signal detector and repeater',
                                     formatter_class=help_formatter)
    parser.add_argument('-s', type=float, required=False, dest='samp_rate',
                        default=7.8128e6, help='Receiver sample rate in SPS')
    parser.add_argument('-t', type=int, required=False, dest='threshold',
                        default=5, help='Detection threshold above noise floor.')
    parser.add_argument('-f', type=float, required=False, dest='freq',
                        default=315e6, help='Receiver tuning frequency in Hz')
    parser.add_argument('-c', type=int, required=False, dest='channel',
                        default=0, help='Receiver channel')
    parser.add_argument('-g', type=str, required=False, dest='rx_gain',
                        default='agc', help='Receive Gain value in dB')
    parser.add_argument('-G', type=float, required=False, dest='tx_gain',
                        default=0, help='Transmit Gain value in dB')
    parser.add_argument('-n', type=int, required=False, dest='buff_len',
                        default=32768, help='Data buffer size (complex samples)')
    return parser.parse_args(sys.argv[1:])


def block_intrlv(x: cp.ndarray, perm: cp.ndarray) -> cp.ndarray:
    # Apply permutation on last axis
    return x[:, perm]

def qam16_mod(bits: cp.ndarray,QAM_16_gain: int) -> cp.ndarray:

    b0 = bits[:, 0::4].astype(cp.int8)  # MSB for I
    b1 = bits[:, 1::4].astype(cp.int8)  # LSB for I
    b2 = bits[:, 2::4].astype(cp.int8)  # MSB for Q
    b3 = bits[:, 3::4].astype(cp.int8)  # LSB for Q


    def gray2level(msb, lsb):
        return 3 - 2*(msb*2 + (msb ^ lsb))

    I = gray2level(b0, b1)
    Q = gray2level(b2, b3)

    symbols = ((I + 1j * Q) / cp.sqrt(10))*QAM_16_gain

    return symbols.astype(cp.complex64)

def qpsk_mod(bits: cp.ndarray,QPSK_gain: int) -> cp.ndarray:

    # Split into even/odd bit streams
    b0 = bits[:, 0::2].astype(cp.int8)  # real axis
    b1 = bits[:, 1::2].astype(cp.int8)  # imag axis

    i = 1 - 2 * b1  # map {0,1} -> {+1,-1}
    q = 1 - 2 * b0

    symbols = ((i + 1j * q) / cp.sqrt(2))*QPSK_gain
    return symbols.astype(cp.complex64)

def add_preamble_guard_postamble(data: cp.ndarray, preamble: cp.ndarray, guard: cp.ndarray, postamble: cp.ndarray,rampup: cp.ndarray,rampdown: cp.ndarray) -> cp.ndarray:
    batch_size, num_hops, sym_per_hop = data.shape
    P = preamble.size if hasattr(preamble, "size") else 1
    G = guard.size if hasattr(guard, "size") else 1
    Q = postamble.size if hasattr(postamble, "size") else 1
    Ru = rampup.size if hasattr(rampup, "size") else 1
    Rd = rampdown.size if hasattr(rampdown, "size") else 1
    hop_out_len = G + P + Ru + sym_per_hop + Q +Rd
    out = cp.zeros((batch_size, num_hops, hop_out_len), dtype=data.dtype)
    # fill with preamble, data, postamble, guard
    out[:, :, 0:G] = guard
    out[:, :, G:G+Ru] = rampup
    out[:, :, G+Ru:G+Ru+P] = preamble
    out[:, :, G+Ru+P:G+Ru+P+sym_per_hop] = data
    out[:, :, G+Ru+P+sym_per_hop:G+Ru+P+sym_per_hop+Q] = postamble
    out[:, :, G+Ru+P+sym_per_hop+Q:] = rampdown
    return out

def upsample_stride(x: cp.ndarray, sps: int) -> cp.ndarray:
    batch_size, num_hops, sym_per_hop = x.shape
    out = cp.zeros((batch_size, num_hops, sym_per_hop * sps), dtype=x.dtype)
    out[:, :, ::sps] = x
    return out

def rcosdesign(beta, span, sps, pulse_type='sqrt'):
    N = span * sps
    t = cp.arange(-N/2, N/2 + 1) / sps
    # Root-raised cosine (RRC)
    h = cp.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 - beta + (4 * beta / cp.pi)
        elif abs(ti) == 1/(4*beta):
            h[i] = (beta / cp.sqrt(2)) * (
                (1 + 2/cp.pi) * cp.sin(cp.pi/(4*beta)) +
                (1 - 2/cp.pi) * cp.cos(cp.pi/(4*beta))
            )
        else:
            h[i] = (
                cp.sin(cp.pi * ti * (1 - beta)) +
                4 * beta * ti * cp.cos(cp.pi * ti * (1 + beta))
            ) / (cp.pi * ti * (1 - (4 * beta * ti)**2))
    return h / cp.sqrt(cp.sum(h**2))  # normalize

def hopwise_fftconvolve(x: cp.ndarray, h: cp.ndarray) -> cp.ndarray:

    batch_size, num_hops, sym_per_hop = x.shape
    x_flat = x.reshape(-1, sym_per_hop)   # (batch_size*num_hops, sym_per_hop)
    h_batch = cp.tile(h, (1, 1))
    # Apply fftconvolve for each row using list comprehension (vectorized at hop level)
    y = fftconvolve(x_flat, h_batch, mode="full") 
    y = y.reshape(batch_size, num_hops, -1)
    return y

def awgn_measured(x: cp.ndarray, snr_db: float, seed: int = None):
    B, H, N = x.shape
    rng = cp.random.RandomState(seed)
     # Compute signal power per hop (mean over samples_per_hop)
    sig_power = cp.mean(cp.abs(x) ** 2, axis=2, keepdims=True).astype(cp.float64)
    # Convert SNR from dB to linear
    snr_lin = 10.0 ** (snr_db / 10.0)
    # Noise variance per hop
    noise_var = sig_power / snr_lin
    # Noise standard deviation
    sigma = cp.sqrt(noise_var / 2.0)
    
    noise_real = rng.normal(0.0, 1.0, (B, H, N)).astype(cp.float32) * cp.sqrt(sigma)
    noise_imag = rng.normal(0.0, 1.0, (B, H, N)).astype(cp.float32) * cp.sqrt(sigma)
    noise = noise_real + 1j * noise_imag
    
    return (x + noise).astype(cp.complex64)

def add_jamming(tx_signal: cp.ndarray,
                jamming_power: cp.ndarray,
                jamming_signal_len: cp.ndarray,
                jammed_idx_lis: cp.ndarray,
                rng: cp.random.RandomState = None) -> cp.ndarray:

    batch_size, num_hops, num_samples = tx_signal.shape

    jam_start_idx_range = (num_samples - jamming_signal_len)
    rand_uniform = cp.random.random(jam_start_idx_range.shape)
    start_idx = (rand_uniform * jam_start_idx_range).astype(cp.int32)    
    # Generate Gaussian jamming signals per hop
    jam_signals = (
        rng.normal(0.0, 1, (batch_size, num_hops, num_samples))* cp.sqrt(jamming_power/2)
        + 1j * rng.normal(0.0, 1, (batch_size, num_hops, num_samples))* cp.sqrt(jamming_power/2)
    ).astype(cp.complex64)

    ## Mask each row to its actual length
    idx = cp.arange(num_samples)[None, None, :]  # shape (1,1,L)
    start = start_idx[:, :, None]                     # (B,H,1)
    end = (start_idx + jamming_signal_len)[:, :, None]           # (B,H,1)
    mask = (idx >= start) & (idx < end)  # boolean mask
    mask = mask.astype(cp.int8)          # convert to 0/1
    jam_signals = jam_signals * mask  
    
    #lengths = jamming_signal_len[:, :, None]
    #mask = cp.arange(jam_signals.shape[2])[None, None, :] < lengths
               
    
    # print(mask.shape)
    # print(jamming_signal_len.shape)
    # print(jam_signals.shape)
    # a = cp.arange(3496)
    # k=1
    # for i in range(5):
    #     for j in range(40):
    #         if jamming_signal_len[i,j] > 0:
    #             print(k,jamming_signal_len[i,j],sum(jam_signals[i,j,:]!=0),cp.max(a[jam_signals[i,j,:]!=0])-cp.min(a[jam_signals[i,j,:]!=0])+1)
    #         k+=1
            
    # Scatter-add into tx_signal
    
    out = tx_signal.copy()
    
    # arange = cp.arange(jam_signals.shape[2])[None, None, :] + start_idx[:, :, None]
    # batch_idx = cp.arange(batch_size)[:, None, None]
    # hop_idx = cp.arange(num_hops)[None, :, None]
    # print(arange.shape)
    # print(arange[0,1,:])
    # print(jam_signals[0,1,:])
    # print(cp.max(arange))
    
    # out[batch_idx, hop_idx, arange] += jam_signals
    # jammed_idx_lis[batch_idx, hop_idx, arange] += jam_signals!=0
    
    out += jam_signals
    jammed_idx_lis+= jam_signals!=0
    return out

def frequency_offset(x: cp.ndarray, fs: float, cfo_hz: cp.ndarray) -> cp.ndarray:
    batch_size, num_hops, samples_per_hop = x.shape
    # Sample indices along last axis
    n = cp.arange(samples_per_hop, dtype=cp.float64)[None, None, :]  # (1,1,S)
    # Phase per sample
    ph = 2.0 * cp.pi * cfo_hz * n / fs  # (B,H,S)
    # Complex exponential rotation
    rot = cp.exp(1j * ph).astype(cp.complex64)  # (B,H,S)
    # Apply rotation
    return (x.astype(cp.complex64, copy=False) * rot)


def adc_scale(x: cp.ndarray, bits_adc: int = 16, adc_backoff: int = 3):
    # x shape: (batch_size, num_samples), dtype complex
    max_r = cp.max(cp.abs(cp.real(x)), axis=1)  # (batch_size,)
    max_i = cp.max(cp.abs(cp.imag(x)), axis=1)  # (batch_size,)
    maximum = cp.maximum(max_r, cp.maximum(max_i, 1e-12))  # (batch_size,)

    actual_bits = cp.where(maximum > 0,
                           cp.round(cp.log2(maximum)) + 1,
                           1).astype(cp.int32)  # (batch_size,)

    scale = 1.0 / (2.0 ** (actual_bits - bits_adc + adc_backoff))  # (batch_size,)

    # reshape scale to (batch_size, 1) for broadcasting
    scale = scale[:, None]

    x_scaled = (x * scale).astype(cp.complex64)
    return x_scaled, scale.astype(cp.float32)




def reduce_input(X: cp.ndarray, Y: cp.ndarray, sps: int):
    batch_size, N = X.shape
    itter = int(cp.ceil(N / sps))
    pad_len = itter * sps - N

    # Pad along time dimension
    if pad_len > 0:
        X = cp.pad(X, ((0, 0), (0, pad_len)), mode="constant")
        Y = cp.pad(Y, ((0, 0), (0, pad_len)), mode="constant")

    # Reshape into chunks: (batch_size, itter, sps)
    X_chunks = X.reshape(batch_size, itter, sps)
    Y_chunks = Y.reshape(batch_size, itter, sps)

    # Find argmax of |X| within each chunk
    abs_chunks = cp.abs(X_chunks)
    idx_in_chunk = cp.argmax(abs_chunks, axis=2)  # (batch_size, itter)

    # Gather values using advanced indexing
    batch_idx = cp.arange(batch_size)[:, None]
    chunk_idx = cp.arange(itter)[None, :]

    X_red = X_chunks[batch_idx, chunk_idx, idx_in_chunk]
    Y_red = Y_chunks[batch_idx, chunk_idx, idx_in_chunk]

    return X_red.astype(cp.complex64), Y_red.astype(cp.float32)

class IQDataGenerator():
    def __init__(self, args,mode):
        if mode == "train":
            self.data_set_size = int(args.train_data_set_size)
        else:
            self.data_set_size = int(args.val_data_set_size)
        self.batch_size = int(args.batch_size)
        
        self.num_hops = int(args.num_hops)
        self.sps = int(args.sps)
        self.fsym = float(args.fsym)
        self.fs = self.sps * self.fsym
        self.codeRate = float(args.codeRate)
        self.rrcFilterSpan = int(args.rrcFilterSpan) 
        
        self.modulationOrderData = int(args.modulationOrderData)
        self.modulationOrderOther = int(args.modulationOrderOther)
        
        self.guardSymLen = int(args.guardSymLen)
        self.rampupSymLen = int(args.rampupSymLen)
        self.preambleSymLen = int(args.preambleSymLen)
        self.dataSymLen = int(args.dataSymLen)
        self.rampdownSymLen = int(args.rampdownSymLen)
        self.postambleSymLen = int(args.postambleSymLen)
        
        self.enBitsPerHop = int(self.dataSymLen*cp.log2(self.modulationOrderData))
        self.uncodedBitsPerHop = int(self.enBitsPerHop*self.codeRate)
        
        self.QPSK_gain = int(args.QPSK_gain)
        self.QAM_16_gain = int(args.QAM_16_gain)
        
        self.tx_hop_signal_len = (self.guardSymLen+self.rampupSymLen+self.preambleSymLen+self.dataSymLen+self.rampdownSymLen+self.postambleSymLen)*self.sps
        
        self.JSRMindB = float(args.JSRMindB)
        self.JSRMaxdB = float(args.JSRMaxdB)
        self.signalPowerdB = float(args.signalPowerdB)
        self.maximum_jamming_pecentage = float(args.maximum_jamming_pecentage)
        self.downsampling = int(args.downsampling)
        
        self.cfo_hz = float(args.cfo_hz)
        
        self.seed = args.seed
        self.rng = cp.random.RandomState(self.seed)
        
        self.hopes_per_input = int(args.hopes_per_input)
        self.CNN_input_size = (self.tx_hop_signal_len // self.downsampling) * self.hopes_per_input
        self.CNN_inputs_per_slot = self.num_hops // self.hopes_per_input
        
        # Load MATLAB files once
        self.preamble = cp.array(loadmat(f"{args.mats_path}preamble.mat")["preamble"].squeeze(), dtype=cp.complex64)[0:self.preambleSymLen]*self.QPSK_gain
        # --- CORRECTED INITIALIZATION ---
        # Initialize postamble as zeros since args.postambleSymLen is 0
        self.postaamble = cp.zeros(self.postambleSymLen, dtype=cp.complex64) 
        self.guard = cp.zeros(self.guardSymLen)
        self.h = cp.array(loadmat(f"{args.mats_path}h.mat")["h"].squeeze(), dtype=cp.float32)
        self.snr_array = cp.array(loadmat(f"{args.mats_path}snr_array.mat")["snr_array"].squeeze(), dtype=cp.float32)
        self.hop_idx_lis = cp.arange(self.tx_hop_signal_len, dtype=cp.int32)
        
        self.ramp_window = cp.concatenate((cp.ones((self.guardSymLen*self.sps)),
                                             cp.hanning(2 * (self.rampupSymLen)*self.sps)[0:(self.rampupSymLen)*self.sps],
                                             cp.ones((self.preambleSymLen+self.dataSymLen)*self.sps),
                                             cp.ones(self.postambleSymLen*self.sps),
                                             cp.hanning(2 * (self.rampdownSymLen)*self.sps)[ (self.rampupSymLen)*self.sps:]))
        
    def __len__(self):
        return int(cp.ceil(self.data_set_size / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[cp.ndarray, cp.ndarray]:
        B = self.batch_size
        tx_len = self.tx_hop_signal_len
        slots_per_batch = B // self.CNN_inputs_per_slot
        max_jammed_hops_per_slot = cp.ceil(self.num_hops*self.maximum_jamming_pecentage)
        rampupBitsLen = int(self.rampupSymLen * cp.log2(self.modulationOrderOther))
        rampdownBitsLen = int(self.rampdownSymLen * cp.log2(self.modulationOrderOther))
        
        X = cp.zeros((B, self.CNN_input_size, 3), dtype=cp.float32)

        jammed_idx_lis = cp.zeros((slots_per_batch,self.num_hops, tx_len), dtype=cp.int8)
        
        batch_seed = None if self.seed is None else int(self.seed + index)
        rng = cp.random.RandomState(batch_seed)

        rbits_up = self.rng.randint(0, 2, (1,rampupBitsLen), dtype=cp.int8)
        rbits_dn = self.rng.randint(0, 2, (1,rampdownBitsLen), dtype=cp.int8)
        rampupsym = qpsk_mod(rbits_up,self.QPSK_gain).astype(cp.complex64)[0,:]
        rampdownsym = qpsk_mod(rbits_dn,self.QPSK_gain).astype(cp.complex64)[0,:]

        snr = rng.choice(self.snr_array,size=slots_per_batch)
        snr = cp.broadcast_to(snr[:, None, None], (slots_per_batch, self.num_hops, 1))
        cfo = cp.array(260)
        cfo = cp.full((slots_per_batch, self.num_hops, 1), cfo)
        
        jammed_hops_per_slot = rng.choice(cp.arange(max_jammed_hops_per_slot+1),slots_per_batch)
        
        perms = cp.argsort(cp.random.rand(slots_per_batch, self.num_hops), axis=1)
        mask = perms < jammed_hops_per_slot[:, None]
        jammed_hops_vec_loc= cp.where(mask, 1, 0)
        
        tx_bits = rng.randint(0, 2,(slots_per_batch, self.num_hops*self.enBitsPerHop), dtype=cp.int8)
        modData = qam16_mod(tx_bits,self.QAM_16_gain)
        modData = modData.reshape(slots_per_batch,self.num_hops,self.dataSymLen)
        mod_frame = add_preamble_guard_postamble(modData, self.preamble, self.guard, self.postaamble,rampupsym,rampdownsym)
        signal_ups = upsample_stride(mod_frame, self.sps)
        signal_tx_fl_rrc = hopwise_fftconvolve(signal_ups,self.h)
        signal_tx_fl_rrc = signal_tx_fl_rrc[:,:,(self.rrcFilterSpan*self.sps)//2:(self.rrcFilterSpan*self.sps)//2+self.tx_hop_signal_len]
        xw = (signal_tx_fl_rrc* self.ramp_window).astype(cp.complex64)
        signal_rx_noisy = awgn_measured(xw, snr, None)
        
        signalPowerLinear = 10**(self.signalPowerdB/10)
        jamming_power_min = signalPowerLinear/(10**(self.JSRMindB/10))
        jamming_power_max = signalPowerLinear/(10**(self.JSRMaxdB/10))
        jamming_power = (jamming_power_min+ (jamming_power_max- jamming_power_min) * cp.random.rand(slots_per_batch, self.num_hops))*jammed_hops_vec_loc
        jamming_power = jamming_power[..., cp.newaxis] 
        jamming_signal_len = rng.randint(1, tx_len + 1,size=(slots_per_batch, self.num_hops))*jammed_hops_vec_loc
    

        jammed_sig = add_jamming(signal_rx_noisy,
                        jamming_power,
                        jamming_signal_len,
                        jammed_idx_lis,
                        rng)
        
        signal_rx_ch = frequency_offset(jammed_sig,cp.array(self.fs),cfo)
        
        signal_rx_ch = signal_rx_ch.reshape(slots_per_batch,self.num_hops*tx_len) 
        jammed_idx_lis = jammed_idx_lis.reshape(slots_per_batch,self.num_hops*tx_len)
        adc_out = signal_rx_ch
        
        
        red_adc_out,red_jammed_idx_lis = reduce_input(adc_out,jammed_idx_lis, self.downsampling) 
        scale = cp.ones((B,self.CNN_input_size)) # Placeholder based on your code
        X[:,:,0] = cp.real(red_adc_out.reshape(B,self.CNN_input_size))
        X[:,:,1] = cp.imag(red_adc_out.reshape(B,self.CNN_input_size))
        X[:,:,2] = scale
        Y = red_jammed_idx_lis.reshape(B,self.CNN_input_size)

        return cp.asnumpy(X),cp.asnumpy(xw)


def tx_task_fn(sdr, tx_stream, tx_sig, tx_buff_len,tx_stop_event):
    """ Transmit task. tx_sig must be an interleaved int16 NumPy array 
        of size 2*tx_buff_len.
    """
    # The SoapySDR writeStream function expects a list of buffers, 
    # where each buffer contains tx_buff_len samples.
    # For SOAPY_SDR_CF32, the samples are complex floats. 
    # However, your current code uses an int16 array, which is common for TX.
    # If the AIR-T is configured to expect int16, this is correct.
    while not tx_stop_event.is_set():
        rc = sdr.writeStream(tx_stream, [tx_sig], tx_buff_len)
        if rc.ret != tx_buff_len:
            raise IOError('Tx Error {}:{}'.format(rc.ret, SoapySDR.errToStr(rc.ret)))
        

def main(args):
    pars = parse_command_line_arguments()

    # --- 1. SDR SETUP ---
    sdr = SoapySDR.Device()
    
    # Configure Receiver/Transmitter
    sdr.setSampleRate(SOAPY_SDR_RX, pars.channel, pars.samp_rate)
    sdr.setGain(SOAPY_SDR_RX, pars.channel, float(pars.rx_gain) if pars.rx_gain.lower() != 'agc' else 30.0) # Set reasonable RX gain
    sdr.setFrequency(SOAPY_SDR_RX, pars.channel, pars.freq)

    sdr.setSampleRate(SOAPY_SDR_TX, pars.channel, pars.samp_rate)
    sdr.setGain(SOAPY_SDR_TX, pars.channel, float(pars.tx_gain) if pars.tx_gain == 0 else pars.tx_gain) # Set TX gain
    sdr.setFrequency(SOAPY_SDR_TX, pars.channel, pars.freq)

    # --- 2. SIGNAL GENERATION ---
    train_generator1 = IQDataGenerator(args,"train")
    X, Y_complex_data = train_generator1.__getitem__(0) # Y is the complex CuPy signal (xw)
    
    # Use the entire generated signal
    tx_sig_cplx = Y_complex_data.reshape(-1)
    tx_samples = tx_sig_cplx.size 
    tx_sig_cplx_np = cp.asnumpy(tx_sig_cplx).astype(np.complex64) 
    
    # --- 3. SYNCHRONIZATION PARAMETERS ---
    # Define scheduled start time 100ms in the future
    start_offset_us = 100000 
    current_time_us = sdr.getHardwareTimeInUsec()
    tx_start_time_us = current_time_us + start_offset_us
    
    # Samples to capture BEFORE transmission starts (e.g., 5000 samples for noise floor)
    pre_tx_samples = 5000 
    
    # Total samples to receive: Pre-TX noise + TX signal length
    total_rx_samples = pre_tx_samples + tx_samples
    
    # --- 4. STREAM SETUP ---
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [pars.channel])
    tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [pars.channel])
    
    # Create the Rx CuPy buffer for the total capture size
    rx_buff_cp = cuda.mapped_array(total_rx_samples, dtype=cp.complex64)

    # --- 5. SCHEDULED ACTIVATION (RX first) ---
    
    # 5a. Activate RX: Start the receiver *before* the TX starts
    # Calculate RX start time to capture pre_tx_samples before TX_start_time
    rx_start_time_us = tx_start_time_us - int((pre_tx_samples / pars.samp_rate) * 1e6)
    
    sdr.activateStream(rx_stream, timeNs=rx_start_time_us * 1000)
    print(f'⏱️ RX scheduled to start at: {rx_start_time_us} us (collecting {pre_tx_samples} pre-TX samples).')
    
    # 5b. Queue TX: Transmit at the defined start time
    print(f'Transmitting {tx_samples} samples at: {tx_start_time_us} us...')
    
    # Queue the TX buffer to be sent at the exact time
    rc = sdr.writeStream(
        tx_stream, 
        [tx_sig_cplx_np], 
        tx_samples, 
        flags=SoapySDR.SOAPY_SDR_HAS_TIME, 
        timeNs=tx_start_time_us * 1000, 
        timeoutUs=5000000
    )
    
    if rc.ret != tx_samples:
        sdr.deactivateStream(tx_stream)
        sdr.deactivateStream(rx_stream)
        sdr.closeStream(tx_stream)
        sdr.closeStream(rx_stream)
        raise IOError(f'Full Tx Error: Only {rc.ret} of {tx_samples} written.')
        
    sdr.deactivateStream(tx_stream)
    print('✅ Transmission scheduled/queued.')
    
    # --- 6. SYNCHRONOUS CAPTURE ---
    
    # Block and wait for the entire reception of (pre_tx_samples + tx_samples)
    print(f'Receiving total {total_rx_samples} complex samples...')
    
    # Set timeout based on expected capture time + margin
    expected_capture_time_us = (total_rx_samples / pars.samp_rate) * 1e6
    timeout_us = int(expected_capture_time_us + 5000000) # 5 seconds margin
    
    sr = sdr.readStream(rx_stream, [rx_buff_cp], total_rx_samples, timeoutUs=timeout_us)

    if sr.ret != total_rx_samples:
        print(f"⚠️ Warning: Received only {sr.ret} of {total_rx_samples} samples. Status: {sr.flags}")
    
    sdr.deactivateStream(rx_stream)
    print('✅ Reception complete.')
    
    # --- 7. DATA PROCESSING AND COMPARISON ---
    rx_signal_complex = cp.asnumpy(rx_buff_cp[:sr.ret])
    
    # Isolate the received signal portion (after the pre-TX samples)
    if sr.ret >= total_rx_samples:
        received_signal_only = rx_signal_complex[pre_tx_samples : pre_tx_samples + tx_samples]
    else:
        # If capture was short, just use the end of the buffer matching TX size
        received_signal_only = rx_signal_complex[-tx_samples:]
    
    # --- MAGNITUDE COMPARISON ---
    print("\n--- Signal Comparison ---")
    tx_mag = np.mean(np.abs(tx_sig_cplx_np)**2)
    rx_mag = np.mean(np.abs(received_signal_only)**2)
    
    if rx_mag > 1e-12:
        power_ratio = tx_mag / rx_mag
        power_loss_db = 10 * np.log10(power_ratio)
        print(f"Tx Mean Power: {10 * np.log10(tx_mag):.2f} dB")
        print(f"Rx Mean Power: {10 * np.log10(rx_mag):.2f} dB")
        print(f"Attenuation/Loss (TX/RX): {power_loss_db:.2f} dB")
    else:
        print("Rx power is near zero. Check gains and physical connection (loopback).")
        
    # --- SAVE COMPLEX OUTPUT ---
    filename = 'received_iq_samples_pretx.txt'
    np.savetxt(filename, rx_signal_complex, fmt='%.8e', delimiter=',', 
               header='Complex IQ Samples (Real, Imag) - Includes pre-TX noise.')
    print(f'💾 Complex data saved to {filename} ({rx_signal_complex.size} total samples).')
    
    # --- 8. PLOT (One Time) ---
    N = rx_signal_complex.size
    fs = pars.samp_rate
    freq = pars.freq
    
    S_np = np.fft.fftshift(np.fft.fft(rx_signal_complex, N) / N)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(111)
    f_ghz = (freq + (np.arange(0, fs, fs/N) - (fs/2) + (fs/N))) / 1e9
    
    plt.plot(f_ghz, 20*np.log10(np.abs(S_np)))
    
    # Highlight the pre-TX noise region
    noise_freq_end = f_ghz[pre_tx_samples * int(N / total_rx_samples)]
    plt.axvline(x=pars.freq / 1e9, color='r', linestyle='--', label='Center Freq')
    plt.axvline(x=noise_freq_end, color='g', linestyle=':', label='End of Pre-TX Noise (Time Domain)')
    
    plt.xlim(f_ghz[0], f_ghz[-1])
    plt.ylim(-100, 0)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Amplitude (dBFS)')
    plt.title(f'Single-Shot Received Spectrum (Scheduled Capture)')
    plt.legend()
    plt.show()

    # --- 9. CLEANUP ---
    sdr.closeStream(rx_stream)
    sdr.closeStream(tx_stream)

    return rx_signal_complex

# (The __main__ block remains the same for argument setup)
    
if __name__ == "__main__":
    import argparse
    # ... (args setup remains the same)
    
    # Simple container class to hold arguments for IQDataGenerator
    class Args:
        pass
    
    args = Args()
    
    args.train_partition_path = "partitions/"
    args.save_path = "final_results/"
    
    args.train_data_set_size = 350000*8
    args.val_data_set_size = int(350000*8*0.2)
    args.batch_size = 144
    args.epochs = 200
    
    args.num_hops = 144 
    args.sps = 4
    args.fsym = 1 / (1.2e-6)
    
    args.codeRate = 3/4
    args.rrcFilterSpan = 40
    
    args.modulationOrderData = 16
    args.modulationOrderOther = 4
    
    args.guardSymLen =1
    args.rampupSymLen =1
    args.preambleSymLen =6
    args.dataSymLen = 4
    args.rampdownSymLen =1
    args.postambleSymLen =0 # IMPORTANT: Set to 0, which prevents slicing error in __init__
    
    args.QPSK_gain = 127*(2**(1/2))
    args.QAM_16_gain = 133.8674
    
    args.JSRMindB = 10
    args.JSRMaxdB = 20
    args.signalPowerdB = 36.7401
    args.maximum_jamming_pecentage = 0.3
    args.downsampling = 4
    
    args.cfo_hz = 260.0
    
    args.hopes_per_input = 72
    
    args.seed = None
    
    args.mats_path = ""
    arch = "arch9_at4"
    method = "IQ_input_at4"
    model_name =f'model_file_{arch}.json'

    # The original call to main(args) expects 'args' to be the result of a parsed command line.
    # Since we are running directly via __main__, we need to manually create the pars object 
    # if any default arguments are needed by other parts of the system.
    
    # NOTE: The arguments for the generator (args) and the SDR setup (pars) are separate.
    # We pass the custom generator 'args' object to main.
    main(args)