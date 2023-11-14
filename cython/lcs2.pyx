# cython: language_level=3, boundscheck=False

import numpy as np
from cython.parallel import parallel, prange
from cython.view cimport array as cvarray
from cython.operator cimport dereference, preincrement

cimport cython
cimport numpy as np
cimport openmp

from libcpp cimport bool as bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from compat import bytes_list_cast, list_of_bytes_list_cast

ctypedef vector[int] int_vec
ctypedef vector[int_vec] int_vec_vec

cdef inline int int_max(int a, int b): return a if a >= b else b

cdef inline int int_max_2(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min_2(int a, int b) nogil: return b if a >= b else a

@cython.boundscheck(False)
def longest_common_subsequence(X, Y):
    """Compute and return the longest common subsequence matrix X, Y are list of strings"""
    cdef int m = len(X)
    cdef int n = len(Y)

    # use numpy array for memory efficiency with long sequences
    # lcs is bounded above by the minimum length of x, y
    assert min(m+1, n+1) < 65535

    #cdef np.ndarray[np.int32_t, ndim=2] C = np.zeros((m+1, n+1), dtype=np.int32)
    cdef np.ndarray[np.uint16_t, ndim=2] C = np.zeros((m+1, n+1), dtype=np.uint16)

    # convert X, Y to C++ standard containers
    cdef vector[string] xx = bytes_list_cast(X)
    cdef vector[string] yy = bytes_list_cast(Y)

    cdef int i, j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if xx[i-1] == yy[j-1]:
                C[i, j] = C[i-1, j-1] + 1
            else:
                C[i, j] = int_max(C[i, j-1], C[i-1, j])
    return C[m,n]


@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline float longest_common_subsequence_2(vector[string]& xx, vector[string]& yy) nogil:
    """Compute and return the longest common subsequence matrix X, Y are list of strings"""
    cdef int m = xx.size()
    cdef int n = yy.size()
    cdef int i, j

    # memoryview to use nogil
    # cdef int [:,:] C = cvarray(shape=(m+1, n+1), itemsize=sizeof(int), format="i")
    cdef int_vec v = int_vec(n+1)
    cdef int_vec_vec C
    C = int_vec_vec(m+1, v)

    for i in range(1, m+1):
        for j in range(1, n+1):
            if xx[i-1] == yy[j-1]:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = int_max_2(C[i][ j-1], C[i-1][ j])
    return (C[m][n] * 2.0) / (m + n)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline float longest_common_subsequence_3(vector[string]& xx, vector[string]& yy) nogil:
    cdef int xx_len = xx.size()
    cdef int yy_len = yy.size()
    cdef int min_len = int_min_2(xx_len, yy_len)
    cdef vector[string] xx_truncated

    cdef int is_hospital = 0
    for j in range(yy_len):
        if yy[j] == b'\xe9\x99\xa2':
            is_hospital = 1

    if is_hospital == 1:
       return longest_common_subsequence_2(xx, yy)

    for i in range(min_len):
        xx_truncated.push_back(xx[i])
    return longest_common_subsequence_2(xx_truncated, yy)

""" X_list is the source and longer names,
    Y_list is the target core_buyer_list, or suspect related companies"""

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_sim_matrix(X_list, Y_list, threshold, key_word_list, head_range_list, head_plus_list, minus_range_list, minus_value_list, plus_range_list, plus_value_list, threads_num=8, match_type="self_match"):
    assert match_type in ["self_match", "core_buyer_match"]
    assert len(key_word_list) == len(head_range_list) == len(head_plus_list) == len(minus_range_list) == len(minus_value_list) == len(plus_range_list) == len(plus_value_list)

    cdef int NUM_THREADS = threads_num

    cdef int m = len(X_list)
    cdef int n = len(Y_list)
    cdef int q = len(key_word_list)
    # since a predefined c-type sim_value can be automatically inferred private, we may not to allocate this giant matrix beforehand
    # cdef vector[vector[float]] C = np.zeros((m, n), dtype=np.float32)
    cdef vector[int] row_inds
    cdef vector[int] col_inds
    cdef vector[float] sim_values

    cdef vector[vector[string]] xxx = list_of_bytes_list_cast(X_list)
    cdef vector[vector[string]] yyy = list_of_bytes_list_cast(Y_list)
    cdef vector[string] key_word_vec = bytes_list_cast(key_word_list)

    cdef vector[int] head_range_vec = head_range_list
    cdef vector[float] head_value_vec = head_plus_list
    cdef vector[int] minus_range_vec = minus_range_list
    cdef vector[float] minus_value_vec = minus_value_list
    cdef vector[int] plus_range_vec = plus_range_list
    cdef vector[float] plus_value_vec = plus_value_list

    cdef float thresh = threshold

    # sim_value in prange will be automatically inferred as thread-local (private)
    # according to http://docs.cython.org/en/latest/src/userguide/parallelism.html
    cdef float sim_value

    cdef int i, j, k

    # init a global lock
    cdef openmp.omp_lock_t lock
    openmp.omp_init_lock(&lock)

    if match_type != "core_buyer_match":
        with cython.nogil, parallel(num_threads=NUM_THREADS):
            for i in prange(m):
                for j in range(n):
                    sim_value = longest_common_subsequence_2(xxx[i], yyy[j])
                    for k in range(q):
                        sim_value = post_process_sim(xxx[i], yyy[j], key_word_vec[k], sim_value, head_range_vec[k], minus_range_vec[k], plus_range_vec[k], head_value_vec[k], minus_value_vec[k], plus_value_vec[k])
                    if threshold_sim(sim_value, thresh):
                        openmp.omp_set_lock(&lock)
                        row_inds.push_back(i)
                        col_inds.push_back(j)
                        sim_values.push_back(sim_value)
                        openmp.omp_unset_lock(&lock)
    else:
        with cython.nogil, parallel(num_threads=NUM_THREADS):
            for i in prange(m):
                for j in range(n):
                    sim_value = longest_common_subsequence_3(xxx[i], yyy[j])
                    for k in range(q):
                        sim_value = post_process_sim(xxx[i], yyy[j], key_word_vec[k], sim_value, head_range_vec[k], minus_range_vec[k], plus_range_vec[k], head_value_vec[k], minus_value_vec[k], plus_value_vec[k])
                    if threshold_sim(sim_value, thresh):
                        openmp.omp_set_lock(&lock)
                        row_inds.push_back(i)
                        col_inds.push_back(j)
                        sim_values.push_back(sim_value)
                        openmp.omp_unset_lock(&lock)
    # destroy global lock
    openmp.omp_destroy_lock(&lock)

    return np.asarray(row_inds), np.asarray(col_inds), np.asarray(sim_values)


cdef extern from "<algorithm>" namespace "std" nogil:
    Iter find[Iter, T](Iter first, Iter last, T pred)

cdef extern from "<algorithm>" namespace "std" nogil:
    T max[T](T a, T b)

cdef extern from "<cmath>" namespace "std" nogil:
    float round(float value)

cdef extern from "<cmath>" namespace "std" nogil:
    int pow(int value, int level)

cdef extern from "<iterator>" namespace "std" nogil:
    vector[string].reverse_iterator make_reverse_iterator(vector[string].iterator it)

""" post-processing for lcs similarity matrix computation """
@cython.boundscheck(False)
cdef inline float post_process_sim(vector[string]& xx, vector[string]& yy, string& key_word, float sim, int head_range=4, int minus_range=4, int plus_range=6, float head_plus=0.0, float minus=0.0, float plus=0.0) nogil:

    cdef vector[string].iterator _xx_iter_ = find(xx.begin(), xx.end(), key_word)
    cdef vector[string].iterator _yy_iter_ = find(yy.begin(), yy.end(), key_word)
    cdef vector[string].reverse_iterator xx_iter = make_reverse_iterator(_xx_iter_)
    cdef vector[string].reverse_iterator yy_iter = make_reverse_iterator(_yy_iter_)

    if ( _xx_iter_ != xx.end() ) and ( _yy_iter_ != yy.end() ):
        minus_same = True
        plus_same = True
        for i in range(1, max(minus_range, plus_range) + 1):
            # to compare the last char
            if xx_iter == xx.rend() and yy_iter == yy.rend():
                break
            if xx_iter == xx.rend() and yy_iter != yy.rend():
                if i <= minus_range: minus_same = False
                if i <= plus_range: plus_same = False
                break
            if xx_iter != xx.rend() and yy_iter == yy.rend():
                if i <= minus_range: minus_same = False
                if i <= plus_range: plus_same = False
                break
            if dereference(xx_iter) != dereference(yy_iter):
                if i <= minus_range: minus_same = False
                if i <= plus_range: plus_same = False
                break
            preincrement(xx_iter)
            preincrement(yy_iter)

        # not same ? minus sim value
        if not minus_same: sim -= minus
        # is same ? add sim value
        if plus_same: sim += plus


    cdef vector[string].iterator xx_iter_ = xx.begin()
    cdef vector[string].iterator yy_iter_ = yy.begin()
    cdef int xx_size = xx.size()
    cdef int yy_size = yy.size()
    if xx_size >= head_range and yy_size >= head_range:
        head_same = True
        for i in range(0, head_range):
            if dereference(xx_iter_) != dereference(yy_iter_):
                head_same = False
                break
            preincrement(xx_iter_)
            preincrement(yy_iter_)
        # if the same head chars, plus sim value
        if head_same: sim += head_plus

    return sim

@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline bool threshold_sim(float sim, float threshold, int precision=6) nogil:
    return round(sim * pow(10, precision)) / pow(10, precision) >= threshold