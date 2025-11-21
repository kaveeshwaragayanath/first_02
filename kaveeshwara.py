from typing import Tuple
import cupy as cp
from scipy.io import loadmat
#from tensorflow import keras
from cupyx.scipy.signal import fftconvolve
import matplotlib.pylab as plt
import sys
import numpy as np
import argparse
import SoapySDR
from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_CS16, errToStr
from SoapySDR import SOAPY_SDR_RX

def convert_to_int16_iq(x):
    """
    Convert complex vector to interleaved int16 IQ format.
    x: complex64 or complex128 numpy array
    """
    a = 2**14
    sig_int16 = np.empty(2 * len(x), dtype=np.int16)
    sig_int16[0::2] = (a * x.real).astype(np.int16)
    sig_int16[1::2] = (a * x.imag).astype(np.int16)
    return sig_int16

def transmit_custom_vec(freq, my_vec, chan=0, fs=31.25e6, gain=-20, tx_len=16384):
    """
    Transmit your own complex vector using AIR-T.
    The vector is zero-padded to tx_len samples.
    """

    # ---------------------------------------------------------
    # 1. Convert your vector to SDR-int16 format
    # ---------------------------------------------------------
    tx_buff = convert_to_int16_iq(my_vec)
    N = len(tx_buff)

    # ---------------------------------------------------------
    # 2. Pad with zeros to reach tx_len
    # ---------------------------------------------------------
    if N < tx_len:
        pad = np.zeros(tx_len - N, dtype=np.int16)
        tx_buff = np.concatenate([tx_buff, pad])
    else:
        tx_buff = tx_buff[:tx_len]

    buff_len = tx_len  # number of int16 samples

    # ---------------------------------------------------------
    # 3. Setup AIR-T SDR
    # ---------------------------------------------------------
    # Baseband vector → TX LO shifts to RF
    lo_freq = freq  # no bb_freq offset since your vector already contains the signal shape

    sdr = SoapySDR.Device()

    sdr.setSampleRate(SOAPY_SDR_TX, chan, fs)
    sdr.setFrequency(SOAPY_SDR_TX, chan, lo_freq)
    sdr.setGain(SOAPY_SDR_TX, chan, gain)

    tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16, [chan])
    sdr.activateStream(tx_stream)

    print("Now Transmitting...   (Ctrl+C to stop)")

    # ---------------------------------------------------------
    # 4. Continuous transmit loop
    # ---------------------------------------------------------
    try:
        while True:
            rc = sdr.writeStream(tx_stream, [tx_buff], buff_len)
            if rc.ret != buff_len:
                print("TX Error {}: {}".format(rc.ret, errToStr(rc.ret)))
    except KeyboardInterrupt:
        pass

    # ---------------------------------------------------------
    # 5. Clean shutdown
    # ---------------------------------------------------------
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)
    print("Stopped.")

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
        self.postaamble = cp.array([])[0:self.postambleSymLen]*self.QPSK_gain
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
        
        X = cp.zeros((B,  self.CNN_input_size, 3), dtype=cp.float32)

        jammed_idx_lis = cp.zeros((slots_per_batch,self.num_hops, tx_len), dtype=cp.int8)
        
        batch_seed = None if self.seed is None else int(self.seed + index)
        rng = cp.random.RandomState(batch_seed)

        rbits_up = self.rng.randint(0, 2, (1,rampupBitsLen), dtype=cp.int8)
        rbits_dn = self.rng.randint(0, 2, (1,rampdownBitsLen), dtype=cp.int8)
        rampupsym = qpsk_mod(rbits_up,self.QPSK_gain).astype(cp.complex64)[0,:]
        rampdownsym = qpsk_mod(rbits_dn,self.QPSK_gain).astype(cp.complex64)[0,:]

        snr = rng.choice(self.snr_array,size=slots_per_batch)
        snr = cp.broadcast_to(snr[:, None, None], (slots_per_batch, self.num_hops, 1))
        cfo = cp.array(260)   # your one element
        cfo = cp.full((slots_per_batch, self.num_hops, 1), cfo)  # fill with the element     
        
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
        #adc_out,scale = adc_scale(signal_rx_ch)    
        adc_out = signal_rx_ch
        
        
        red_adc_out,red_jammed_idx_lis = reduce_input(adc_out,jammed_idx_lis, self.downsampling)  
        #scale = cp.repeat(scale, self.CNN_inputs_per_slot, axis=0)[:,0]
        #scale = cp.broadcast_to(scale[:, None], (B, self.CNN_input_size))  
        scale = cp.ones((144,936))
        X[:,:,0] = cp.real(red_adc_out.reshape(B,self.CNN_input_size))
        X[:,:,1] = cp.imag(red_adc_out.reshape(B,self.CNN_input_size))
        X[:,:,2] = scale
        Y = red_jammed_idx_lis.reshape(B,self.CNN_input_size)

        # #return cp.asnumpy(X), cp.asnumpy(Y),cp.asnumpy(snr)
        return cp.asnumpy(mod_frame),cp.asnumpy(xw)
    
    
if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description = 'Train and validation pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.train_partition_path =  "partitions/"
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
    args.postambleSymLen =0
    
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

    train_generator1 = IQDataGenerator(args,"train")
    X,Y = train_generator1.__getitem__(0)


    tx_chan = 0
    tx_fs   = 31.25e6
    tx_gain = -20
    tx_freq = 2.40e9
    TX_LEN  = 16384                # AIR-T recommended TX buffer size

###############################################################################
# RX SETTINGS
###############################################################################
    rx_chan = 0
    N       = 16384                # Receive same as TX buffer
    rx_fs   = 31.25e6
    rx_freq = 2.40e9
    use_agc = True
    timeout_us = int(5e6)
    rx_bits    = 16


###############################################################################
# Prepare transmit buffer
###############################################################################
    tx_raw = convert_to_int16_iq(Y)

# Zero-pad to TX_LEN
    if len(tx_raw) < TX_LEN:
        pad = np.zeros(TX_LEN - len(tx_raw), dtype=np.int16)
        tx_raw = np.concatenate([tx_raw, pad])
    else:
        tx_raw = tx_raw[:TX_LEN]

    print("TX buffer prepared:", len(tx_raw), "samples int16")

###############################################################################
# Create SDR object
###############################################################################
    sdr = SoapySDR.Device()

###############################################################################
# TX SETUP
###############################################################################
    sdr.setSampleRate(SOAPY_SDR_TX, tx_chan, tx_fs)
    sdr.setFrequency  (SOAPY_SDR_TX, tx_chan, tx_freq)
    sdr.setGain       (SOAPY_SDR_TX, tx_chan, tx_gain)

    tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CS16, [tx_chan])
    sdr.activateStream(tx_stream)

###############################################################################
# RX SETUP
###############################################################################
    sdr.setSampleRate(SOAPY_SDR_RX, rx_chan, rx_fs)
    sdr.setFrequency  (SOAPY_SDR_RX, rx_chan, rx_freq)
    sdr.setGainMode   (SOAPY_SDR_RX, rx_chan, use_agc)

    rx_buff   = np.empty(2*N, np.int16)
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan])
    sdr.activateStream(rx_stream)

###############################################################################
# START TX (continuous) + RX (single block)
###############################################################################
    print("\nNow Transmitting + Receiving...")

# --- TX one buffer while simultaneously capturing RX block ---
    rc_tx = sdr.writeStream(tx_stream, [tx_raw], TX_LEN)
    if rc_tx.ret != TX_LEN:
        print("TX Error:", rc_tx.ret, errToStr(rc_tx.ret))

# --- Receive block ---
    sr = sdr.readStream(rx_stream, [rx_buff], N, timeoutUs=timeout_us)
    rc_rx = sr.ret
    assert rc_rx == N, f"RX error: {rc_rx}"

    print("RX received samples:", rc_rx)

###############################################################################
# STOP STREAMS
###############################################################################
    sdr.deactivateStream(tx_stream)
    sdr.closeStream(tx_stream)

    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)

    print("Stopped TX/RX.")

###############################################################################
# CONVERT RX DATA TO COMPLEX
###############################################################################
    s0 = rx_buff.astype(float) / (2**(rx_bits-1))
    s  = s0[::2] + 1j*s0[1::2]

###############################################################################
# FFT of received signal
###############################################################################
    S = np.fft.fftshift(np.fft.fft(s) / N)

###############################################################################
# PLOT RESULTS
###############################################################################
    plt.figure(figsize=(13,8))

# Time domain
    plt.subplot(2,1,1)
    t_us = np.arange(N) / rx_fs / 1e-6
    plt.plot(t_us, s.real, label='I')
    plt.plot(t_us, s.imag, label='Q')
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Received I/Q Time Domain")

# Frequency domain
    plt.subplot(2,1,2)
    freq_axis = (np.arange(N) - N//2) * (rx_fs / N)
    plt.plot(freq_axis/1e6, 20*np.log10(np.abs(S)))
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Received Spectrum")

    plt.tight_layout()
    plt.show()
    transmit_custom_vec(freq=2400e6, my_vec=Y)
    

