cimport numpy as cnp
import numpy as np
from scipy import sparse

cdef tuple sort_csr_matrix_rowwise(cnp.int32_t[:] indptr,
                                    cnp.int32_t[:] indices,
                                    cnp.float64_t[:] data,
                                    cnp.int32_t order):
    cdef cnp.int32_t[:] sorted_indices = np.empty_like(indices)
    cdef cnp.float64_t[:] sorted_data = np.empty_like(data)
    
    cdef list argsort
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    
    for i in range(indptr.shape[0]-1):  # Iterate over all nodes
        argsort = list(np.argsort(data[indptr[i]:indptr[i+1]]) + indptr[i])[::order]
        for j in range(len(argsort)):
            #print(f'i: {i}, j: {j}, argsort[j]: {argsort[j]}, data[argsort[j]]: 
            #{data[argsort[j]]}, j + indptr[i]: {j + indptr[i]}')
            sorted_data[j + indptr[i]] = data[argsort[j]]
            sorted_indices[j + indptr[i]] = indices[argsort[j]]
    
    return indptr, sorted_indices, sorted_data

cpdef sort_L_cython(L, ascending_descending='ascending'):
    L_csr = sparse.csr_matrix(L)
    
    L_indptr = L_csr.indptr
    L_indices = L_csr.indices
    L_data = L_csr.data  # Payables
    
    cdef cnp.int32_t order
    
    # Descending or ascending
    if ascending_descending=='descending':
        order = -1
    elif ascending_descending=='ascending':
        order = 1
    else:
        raise Exception(f'Wrong order-way ({ascending_descending})')
    
    print('sorting L')
    new_L_indptr, new_L_indices, new_L_data = sort_csr_matrix_rowwise(L_indptr, L_indices, L_data, order)
    new_L = sparse.csr_matrix((new_L_data, new_L_indices, new_L_indptr), shape=L_csr.shape)
    print('Done sorting L')

    return new_L