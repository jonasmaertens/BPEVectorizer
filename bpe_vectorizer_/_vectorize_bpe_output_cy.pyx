import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

def _vectorize_bpe_output_cy(bpe_res):
    cdef vector[vector[int]] bpe_res_v = bpe_res
    cdef int i, j, k, num, idx
    cdef vector[int] doc
    cdef size_t num_docs = len(bpe_res)
    with nogil:
        num = 0
        for j in range(num_docs):
            num += bpe_res_v[j].size()
    data = np.ones(num, dtype=np.intc)
    indices = np.zeros(num, dtype=np.intc)
    indptr = np.zeros(len(bpe_res) + 1, dtype=np.intc)
    cdef int[:] indices_v = indices
    cdef int[:] indptr_v = indptr
    cdef int[:] data_v = data
    i = 0
    with nogil:
        for j in range(num_docs):
            for k in range(bpe_res_v[j].size()):
                indices_v[i] = bpe_res_v[j][k]
                i += 1
            indptr_v[j + 1] = i

    return data, indices, indptr