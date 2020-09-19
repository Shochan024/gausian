#!-*-coding:utf-8-*-
import numpy as np

__all__ = ["RBF2D"]

class RBF2D:
    def __init__( self , *param ):
        self.param = list( param )

    def __call__( self , x1 , x2 ):
        a,s,w = self.param

        return a ** 2 * np.exp( -( ( x1 - x2 ) / s ) ** 2 ) + w * (  x1 == x2 )
