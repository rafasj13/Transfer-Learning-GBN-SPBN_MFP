# ctypes (interfaz con C)
from ctypes import POINTER, c_double, c_long, cast
import ctypes as ct
from numpy.ctypeslib import load_library
import numpy as np

DOUBLE = ct.c_double
PDOUBLE = ct.POINTER(DOUBLE)
PPDOUBLE = ct.POINTER(PDOUBLE)

class MyStruct(ct.Structure):
    _fields_ = [
        ('X', ct.POINTER(ct.c_double)),
        ('Z', PPDOUBLE)
    ]

def double2ArrayToPointer(arr):
    """ Converts a 2D numpy to ctypes 2D array. 
    
    Arguments:
        arr: [ndarray] 2D numpy float64 array

    Return:
        arr_ptr: [ctypes double pointer]

    """

    # Init needed data types
    ARR_DIMX = DOUBLE*arr.shape[1]
    ARR_DIMY = PDOUBLE*arr.shape[0]

    # Init pointer
    arr_ptr = ARR_DIMY()

    # Fill the 2D ctypes array with values
    for i, row in enumerate(arr):
        arr_ptr[i] = ARR_DIMX(*row)

    return arr_ptr

def fastResiduals(X, Z, bandwidth):
  

    ROWS = X.shape[0]
    ZCOLS = Z.shape[1]
    residuals = np.zeros_like(X)

    mystruct = MyStruct()
    mystruct.X = (c_double * X.shape[0])(*X)  
    mystruct.Z = double2ArrayToPointer(Z.astype(np.float64))
    residuals_c = (c_double * residuals.shape[0])(*residuals)


    residualsC = load_library('residuals.so', './code/wrapper/')
    residualsC.fastResiduals.argtypes = [POINTER(MyStruct), c_long, c_long, c_double,
                                    POINTER(c_double)]

    residualsC.fastResiduals(mystruct, ROWS, ZCOLS, bandwidth, residuals_c)

    residuals = np.ctypeslib.as_array(cast(residuals_c, POINTER(c_double)), shape=(residuals.shape[0],))
    return residuals

