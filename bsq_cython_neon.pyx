# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""Apple Silicon NEON fast path used by `view.py` (BSQ RGB -> BGR HWC)."""

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as np

cdef extern from "arm_neon.h":
    ctypedef struct uint8x16_t:
        pass

    ctypedef struct uint8x16x3_t:
        uint8x16_t val[3]

    uint8x16_t vld1q_u8(const unsigned char *ptr) nogil
    void vst3q_u8(unsigned char *ptr, uint8x16x3_t val) nogil


cdef extern from "arm_acle.h":
    void __pld(const void *addr) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bsq_to_bgr_neon_ultimate(np.uint8_t[:, :, ::1] bsq):
    if bsq.shape[0] != 3:
        raise ValueError(f"expected bsq shape (3, H, W), got {tuple(bsq.shape)}")

    cdef int height = bsq.shape[1]
    cdef int width = bsq.shape[2]
    cdef int width_vec = (width // 16) * 16
    cdef int h, w, w_vec

    cdef np.ndarray[np.uint8_t, ndim=3] bgr = np.empty((height, width, 3), dtype=np.uint8)

    cdef unsigned char *r_ptr
    cdef unsigned char *g_ptr
    cdef unsigned char *b_ptr
    cdef unsigned char *out_ptr

    cdef uint8x16_t vr, vg, vb
    cdef uint8x16x3_t interleaved

    with nogil:
        for h in prange(height, schedule="static", num_threads=8):
            if h + 2 < height:
                __pld(&bsq[0, h + 2, 0])
                __pld(&bsq[1, h + 2, 0])
                __pld(&bsq[2, h + 2, 0])

            for w_vec in range(0, width_vec, 16):
                if w_vec + 128 < width:
                    __pld(&bsq[0, h, w_vec + 128])
                    __pld(&bsq[1, h, w_vec + 128])
                    __pld(&bsq[2, h, w_vec + 128])

                r_ptr = &bsq[0, h, w_vec]
                g_ptr = &bsq[1, h, w_vec]
                b_ptr = &bsq[2, h, w_vec]

                vr = vld1q_u8(r_ptr)
                vg = vld1q_u8(g_ptr)
                vb = vld1q_u8(b_ptr)

                interleaved.val[0] = vb
                interleaved.val[1] = vg
                interleaved.val[2] = vr

                out_ptr = &bgr[h, w_vec, 0]
                vst3q_u8(out_ptr, interleaved)

            for w in range(width_vec, width):
                bgr[h, w, 0] = bsq[2, h, w]
                bgr[h, w, 1] = bsq[1, h, w]
                bgr[h, w, 2] = bsq[0, h, w]

    return bgr
