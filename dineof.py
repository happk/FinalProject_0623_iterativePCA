import numpy as np
import scipy.sparse.linalg as spla
from sklearn.utils.extmath import randomized_svd

def dineof(Xo, n_max=None, ref_pos=None, delta_rms=1e-5, method="svds"):
    if n_max is None:
        n_max = Xo.shape[1]
    
    na_true = np.isnan(Xo)
    na_false = ~na_true
    
    if ref_pos is None:
        valid_indices = np.where(na_false.flatten())[0]
        ref_pos = np.random.choice(valid_indices, min(max(30, int(0.01 * na_false.sum())), len(valid_indices)), replace=False)
    
    Xa = Xo.copy()
    Xa_flat = Xa.flatten()
    Xa_flat[ref_pos] = 0
    Xa = Xa_flat.reshape(Xo.shape)
    Xa[na_true] = 0
    
    rms_prev = np.inf
    rms_now = np.sqrt(np.mean((Xa.flatten()[ref_pos] - Xo.flatten()[ref_pos])**2))
    n_eof = 1
    
    RMS = [rms_now]
    NEOF = [n_eof]
    Xa_best = Xa.copy()
    n_eof_best = n_eof
    
    while rms_prev - rms_now > delta_rms and n_eof < n_max:
        while rms_prev - rms_now > delta_rms:
            rms_prev = rms_now
            
            if method == "svds":
                u, s, vt = spla.svds(Xa, k=n_eof)
            elif method == "randomized":
                u, s, vt = randomized_svd(Xa, n_components=n_eof)
            else:
                raise ValueError("Unknown method: choose 'svds' or 'randomized'")
            
            RECi = np.dot(np.dot(u, np.diag(s)), vt)
            RECi_flat = RECi.flatten()
            Xa_flat[ref_pos] = RECi_flat[ref_pos]
            Xa_flat[na_true.flatten()] = RECi_flat[na_true.flatten()]
            Xa = Xa_flat.reshape(Xo.shape)
            
            rms_now = np.sqrt(np.mean((Xa.flatten()[ref_pos] - Xo.flatten()[ref_pos])**2))
            print(f"{n_eof} EOF; RMS = {rms_now:.8f}")
            
            RMS.append(rms_now)
            NEOF.append(n_eof)
            
            if rms_now == min(RMS):
                Xa_best = Xa.copy()
                n_eof_best = n_eof
        
        n_eof += 1
        rms_prev = rms_now
        
        if method == "svds":
            u, s, vt = spla.svds(Xa, k=n_eof)
        elif method == "randomized":
            u, s, vt = randomized_svd(Xa, n_components=n_eof)
        else:
            raise ValueError("Unknown method: choose 'svds' or 'randomized'")
        
        RECi = np.dot(np.dot(u, np.diag(s)), vt)
        RECi_flat = RECi.flatten()
        Xa_flat[ref_pos] = RECi_flat[ref_pos]
        Xa_flat[na_true.flatten()] = RECi_flat[na_true.flatten()]
        Xa = Xa_flat.reshape(Xo.shape)
        
        rms_now = np.sqrt(np.mean((Xa.flatten()[ref_pos] - Xo.flatten()[ref_pos])**2))
        print(f"{n_eof} EOF; RMS = {rms_now:.8f}")
        
        RMS.append(rms_now)
        NEOF.append(n_eof)
        
        if rms_now == min(RMS):
            Xa_best = Xa.copy()
            n_eof_best = n_eof
    
    Xa = Xa_best
    n_eof = n_eof_best
    Xa_flat = Xa.flatten()
    Xa_flat[ref_pos] = Xo.flatten()[ref_pos]
    Xa = Xa_flat.reshape(Xo.shape)
    
    result = {
        'Xa': Xa,
        'n_eof': n_eof,
        'RMS': RMS,
        'NEOF': NEOF,
        'ref_pos': ref_pos
    }
    
    return result