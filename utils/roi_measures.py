import numpy as np

def mad(A, B) :
    intersect = np.multiply(A != 0, B != 0)
    max_a = A[intersect].max()
    max_b = B[intersect].max()
    return np.median(np.abs(A[intersect] / max_a - B[intersect] / max_b))

def ssim(A_, B_) :
    A = A_ / A_.max()
    B = B_ / B_.max()
    intersect = np.multiply(A != 0, B != 0)
    ua = A[intersect].mean()
    ub = B[intersect].mean()
    oa = A[intersect].std() ** 2
    ob = B[intersect].std() ** 2
    oab = np.sum(np.multiply(A[intersect] - ua, B[intersect] - ub)) / (np.sum(intersect) - 1)
    k1 = 0.01
    k2 = 0.03
    L = 1
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    num = (2*ua*ub + c1) * (2*oab + c2)
    den = (ua**2 + ub**2 + c1) * (oa + ob + c2)
    return num / den