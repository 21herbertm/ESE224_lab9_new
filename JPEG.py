#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:02:18 2021

@author: zhiyangwang
"""

import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

class DCT_2D():
    """
    2-D DCT
    """
    def __init__(self, x):
        """
        input time-domain signal x
        """
        self.x = x
        self.M = np.shape(x)[0]
        self.N = np.shape(x)[1]
    
    def solve2(self):
        """
        \\\\\ METHOD: Compute DCT of x with built-in function
        """
        return dct(dct(self.x.T, norm='ortho').T, norm='ortho')
    

class iDCT_2D():
    """
    2-D iDCT
    """
    
    def __init__(self, X):
        self.X = X
    
    def solve2(self):
        """
        \\\\\ METHOD: Compute iDCT of X with built-in function
        """
        return idct(idct(self.X.T, norm='ortho').T, norm='ortho')

def quantization_table(quality):
    Q50 = [ [16, 11, 10, 16, 24, 40, 51, 61],
           [12, 12, 14, 19, 26, 58, 60, 55],
    [ 14, 13, 16, 24, 40, 57, 69, 56],
    [ 14, 17, 22, 29, 51, 87, 80, 62],
    [ 18, 22, 37, 56, 68, 109, 103, 77],
    [ 24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]]
    
    QX = np.zeros((8, 8))
    if quality > 50:
        QX =( Q50 * (np.ones((8,8)) * ( ( 100 - quality ) / 50 ) )).round().astype(np.int32) 
    else:
        QX = (Q50 *( np.ones((8,8)) * ( 50 / quality))).round().astype(np.int32)
         
    return QX



def quantize(QX, block):
       
    return (block /QX).round().astype(np.int32)



def dequantize(QX, block):
         
    return (( block) * QX).astype(np.int32)
    





def q13(quantization):
    img = mpimg.imread('imgB_prenoise.png')  
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()
    N = img.shape[0]
    X_quantized = np.zeros([N, N], dtype=np.complex)
    for i in np.arange(0, N-7, 8):
        for j in np.arange(0, N-7, 8):
            x_block = img[ i : i + 8, j : j + 8 ] * 256
            block_DCT = DCT_2D(x_block)
            X_block = block_DCT.solve2()
            quant_block = quantize(quantization, X_block)
            X_quantized[i : i + 8, j : j + 8 ] = quant_block
            
    return X_quantized


def q14_2(quantization, X_quan):
    N = X_quan.shape[0]
    x_dequantized = np.zeros([N, N], dtype=np.complex)
    for i in np.arange(0, N-7, 8):
        for j in np.arange(0, N-7, 8):
            X_block = X_quan[ i : i + 8, j : j + 8 ]
            X_dequan_block = dequantize(quantization, X_block)
            block_iDCT = iDCT_2D(X_dequan_block)
            x_block = block_iDCT.solve2()/256
            x_dequantized[i : i + 8, j : j + 8 ] = x_block
    energy_rec = np.amax(x_dequantized)
    x_dequantized /= energy_rec
            
    return x_dequantized
    
###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__': 
    # JPEG with the given quantization matrix
    img = mpimg.imread('imgB_prenoise.png')  

    quan_tab = quantization_table(50)            
    X = q13(quan_tab)
    x = q14_2(quan_tab, X) 
    plt.imshow(abs(x), cmap='gray')
    plt.title('Reconstructed image with standard quantization matrix')
    plt.savefig('Recons_jpeg.png')
    print(np.linalg.norm(img - abs(x), 2))
    plt.colorbar()
    plt.show()    
    
    # Compress and reconstruct with a lower value quantization matrix
    quan_tab = quantization_table(95)      
    print(quan_tab)      
    X = q13(quan_tab)
    x = q14_2(quan_tab, X) 
    plt.imshow(abs(x), cmap='gray')
    print(np.linalg.norm(img - abs(x), 2))

    plt.title('Reconstructed image with lower value quantization matrix')
    plt.savefig('Reconslow_jpeg.png')

    plt.colorbar()
    plt.show()  
    
    # Compress and reconstruct with a higher value quantization matrix
    quan_tab = quantization_table(2)   
    print(quan_tab)      
         
    X = q13(quan_tab)
    x = q14_2(quan_tab, X) 
    plt.imshow(abs(x), cmap='gray')
    print(np.linalg.norm(img - abs(x), 2))

    plt.title('Reconstructed image with higher value quantization matrix')
    plt.savefig('Reconshigh_jpeg.png')

    plt.colorbar()
    plt.show()  


   



