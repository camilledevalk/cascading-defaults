cimport numpy as cnp
import numpy as np
import scipy.sparse as sparse

cdef cnp.float64_t[:] payments_robin_hood(cnp.int32_t[:] L_indptr,
                                          cnp.int32_t[:] L_indices,
                                          cnp.float64_t[:] L_data,
                                          cnp.float64_t[:] total_payables,
                                          cnp.float64_t[:] incomings,
                                          cnp.float64_t[:] equities,
                                          cnp.int32_t order,
                                          bint pay_remaining_money):
    
    cdef cnp.float64_t[:] amounts_payed = np.zeros_like(L_data)
    cdef cnp.float64_t total_amount_payed = 0.
    cdef cnp.float64_t[:] equities_creditors
    cdef cnp.float64_t[:] zeros = np.zeros_like(L_indptr, dtype=np.float64)
    cdef cnp.int32_t[:] creditors
    cdef cnp.int32_t[:] argsort
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    
    # Order is defined in general the other way around, thus flip it in this strategy
    order = -1 * order
    
    # Step 4
    # For all companies calculate all the payments
    for i in range(L_indptr.shape[0]-1):  # Iterate over all nodes  
        total_amount_payed = 0.
        # Step 4.1
        # Check if a company can pay all its creditors (i.e. total_payables <= total_incoming)
        # If yes: pay all creditors
        if incomings[i] >= total_payables[i]:  # If they get more then they have to pay
            amounts_payed[L_indptr[i]: L_indptr[i+1]] = L_data[L_indptr[i]: L_indptr[i+1]]  # Copy all data from row in p
        # If not: step 4.2
        else:
            # Step 4.2 Robin Hood
            # Step 4.2.1
            # Argsort equities[creditors]
            creditors = L_indices[L_indptr[i]: L_indptr[i+1]]
            equities_creditors = zeros[:len(creditors)]
            for j in range(len(creditors)):
                equities_creditors[j] = equities[creditors[j]]
            argsort = np.array(np.argsort(equities_creditors)[::order] + L_indptr[i], dtype=np.int32)
            # Pay edge with equity[creditor] is smallest until total_payments >= total_incoming
            for j in range(len(argsort)):
                if total_amount_payed + L_data[argsort[j]] < incomings[i]:
                    assert (not amounts_payed[argsort[j]])
                    amounts_payed[argsort[j]] = L_data[argsort[j]]
                    total_amount_payed += L_data[argsort[j]]
                elif pay_remaining_money != 0:
                    amounts_payed[argsort[j]] = incomings[i] - total_amount_payed
                    break
                else:
                    break
            
    return amounts_payed
    

cdef cnp.float64_t[:] payments_largest_creditor(cnp.int32_t[:] L_indptr,
                                                cnp.int32_t[:] L_indices,
                                                cnp.float64_t[:] L_data,
                                                cnp.float64_t[:] total_payables,
                                                cnp.float64_t[:] incomings,
                                                bint pay_remaining_money):
    
    cdef cnp.float64_t[:] amounts_payed = np.zeros_like(L_data)
    cdef cnp.float64_t total_amount_payed = 0.
    cdef Py_ssize_t i
    cdef Py_ssize_t j
        
    # Step 4
    # For all companies calculate all the payments
    for i in range(L_indptr.shape[0]-1):  # Iterate over all nodes  
        total_amount_payed = 0.
        # Step 4.1
        # Check if a company can pay all its creditors (i.e. total_payables <= total_incoming)
        # If yes: pay all creditors
        if incomings[i] >= total_payables[i]:  # If they get more then they have to pay
            amounts_payed[L_indptr[i]: L_indptr[i+1]] = L_data[L_indptr[i]: L_indptr[i+1]]  # Copy all data from row in p
        # If not: step 4.2
        else:        
            # Step 4.2
            # Pay smallest payable until total_payments >= total_incoming
            # BTW, This only works because the data was sorted
            for j in range(L_indptr[i], L_indptr[i+1]):
                if total_amount_payed + L_data[j] < incomings[i]:
                    amounts_payed[j] = L_data[j]
                    total_amount_payed += L_data[j]
                elif pay_remaining_money != 0:
                    amounts_payed[j] = incomings[i] - total_amount_payed
                    break
                else:
                    break
    
    return amounts_payed

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
            sorted_data[j + indptr[i]] = data[argsort[j]]
            sorted_indices[j + indptr[i]] = indices[argsort[j]]
    
    return indptr, sorted_indices, sorted_data
        

cpdef cython_payments(cls, strategy='largest_creditor', last_first='first', pay_remaining_money=False):
    """
    Icnputs: self (FictitiousDefaultsAlgorithm)
    Output: a CSR-matrix P_ij with the payments made by company i to j, where company i
    doesn't pay it's largest creditor(s) when it can't
    """
    possible_cython_strategies = ['largest_creditor', 'robin_hood']
    
    assert strategy in possible_cython_strategies, f'Strategy {strategy} not implemented.'
    
    # Step 1
    # Redefine icnputs to let Cython accept types
    p_csr = sparse.csr_matrix(cls.p)
    L_csr = sparse.csr_matrix(cls.L)
    
    p_indptr = p_csr.indptr
    p_indices = p_csr.indices
    p_data = p_csr.data  # Payments
    
    L_indptr = L_csr.indptr
    L_indices = L_csr.indices
    L_data = L_csr.data  # Payables
    
    cdef bint pay_remaining_money_c
    pay_remaining_money_c = 1 if pay_remaining_money else 0
    
    cdef cnp.float64_t[:] equities_view
    if strategy == 'robin_hood':
        equities_view = cls.equities
    
    # Step 1.2
    # Sort L if needed
    cdef cnp.int32_t order
    
    # Descending or ascending
    if last_first=='first':
        order = -1
    elif last_first=='last':
        order = 1
    else:
        raise Exception(f'Wrong order-way ({last_first})')
    
    # Step 2
    # Calculate total_payables of all companies
    cdef cnp.float64_t[:] total_payables_view = cls.total_payables_array
    # Step 3
    # Calculate total_incoming of all companies
    cdef cnp.float64_t[:] incomings_view = cls.total_incoming
    
    cdef cnp.float64_t[:] amounts_payed_view
    # Step 4
    # For all companies calculate all the payments
    if strategy == 'largest_creditor':
        amounts_payed_view = payments_largest_creditor(L_indptr, L_indices, L_data,
                                                       total_payables_view, incomings_view, pay_remaining_money_c)
    elif strategy == 'robin_hood':
        amounts_payed_view = payments_robin_hood(L_indptr, L_indices, L_data, total_payables_view,
                                                  incomings_view, equities_view, order, pay_remaining_money_c)
            
    # Step 5
    # Aggregate all payments (rows) to one payment matrix
    new_p = sparse.csr_matrix((amounts_payed_view, L_indices, L_indptr), shape=p_csr.shape)
    return new_p

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