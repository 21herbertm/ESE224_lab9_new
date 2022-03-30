#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 22:46:01 2021

@author: zhiyangwang
"""

import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from scipy.fft import dct, idct

import discrete_signal


class inner_prod_2D():
    """
    2—D inner-product
    """
    def __init__(self, x, y):
        """
        x,y: two 2-D signals
        """
        self.x=x
        self.y=y
        self.N=np.shape(x)[0]

    def solve(self):
        """
        \\\\\ METHOD: Compute the inner product
        """
        prod = 0        
        for i in range(self.N):
            for j in range(self.N):
                prod = prod + self.x[i,j] * np.conj(self.y[i,j])
            
        return prod
    
class Generate_Discrete_Cosine_2D(object):
    """
    Creates a discrete cosine matrix of discrete frequency k and l, with duration M and N.
    Arguments:
        k l: discrete frequency
        M N: duration of the cosine matrix
    """

    def __init__(self, k, l, M, N):
        self.k = k
        self.l = l
        self.M = M
        self.N = N

        # Vector containing elements of time indexes
        self.n = np.arange(N)
        self.m = np.arange(M)

        # Vector containing elements of the complex exponential
        self.dis_cos_2d = np.matmul( np.cos(self.k * cmath.pi / 2/self.M * (2*self.m+1) ).reshape((self.M,1)) ,np.cos(self.l * cmath.pi / 2/self.N * (2*self.n+1)).reshape((1,self.N)) )
        

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

    def solve(self):
        """
        \\\\\ METHOD: Compute DCT of x
        """
        X = np.zeros([self.M, self.N], dtype=np.complex)
        for k in range(self.M):
            if k == 0:
                c1 = 1/np.sqrt(2)
            else:
                c1 = 1
                
            for l in range(self.N):
                if l == 0:
                    c2 = 1 / np.sqrt(2)
                else:
                    c2 = 1
                    
                twod_cos = Generate_Discrete_Cosine_2D(k,l,self.M, self.N).dis_cos_2d
                inner_prod = inner_prod_2D(self.x, twod_cos)
                X[k, l] =  2 / np.sqrt(self.M)/ np.sqrt(self.N) *c1 *c2 * inner_prod.solve()
        return X

    
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
    
    

class DFT_2D():
    """
    2-D DFT
    """
    def __init__(self, x):
        """
        input time-domain signal x
        """
        self.x=x
        self.M=np.shape(x)[0]
        self.N=np.shape(x)[1]

    def solve(self):
        """
        \\\\\ METHOD: Compute 2D-DFT of x
        """
        X=np.zeros([self.M, self.N], dtype=np.complex)
        for m in range(self.M):
            for n in range(self.N):
                for i in range(self.M):
                    for j in range(self.N):
                        X[m,n] = X[m,n] + self.x[i,j]/np.sqrt(self.M*self.N)*np.exp(-1j*2*cmath.pi*(m*i/self.M+n*j/self.N))
            
        return X
    def solve2(self):
        """
        \\\\\ METHOD: Compute 2D-DFT of x with built-in function
        """
        X = np.fft.fft2(self.x) / np.sqrt(self.M*self.N)
        X_shift = np.fft.fftshift(X)

        return X_shift


def compress_block(X, K):
    """
    Compress the block by keeping the K-largest coefficients of the input X block 
    X is an 8*8 matrix with DFT or DCT coefficients
    """  
    X_truncated = np.zeros([8, 8], dtype=np.complex)
    E = np.zeros(64)
    E = abs( X.reshape((1, 64) ))
    index_temp = np.argsort(-E)
    index = np.array(index_temp[:, 0: K])[0]
    for i in np.arange(len(index)):
        index_x = np.int( np.floor(index[i] / 8))
        index_y = np.int( index[i] - index_x * 8)
        # Keep the K-largest coefficients and set the rest as 0
        X_truncated[index_x, index_y] = X[index_x, index_y]    
    # return the compressed X block and the index for reserved frequency
    return X_truncated, index
    


def compress_image_DCT(x, K):
    """
    Partition the image into 8*8 blocks, compute the DCT of each block and compress
    """ 
    N = x.shape[0]
    
    X_compressed = np.zeros([N, N], dtype=np.complex)
    for i in np.arange(0, N+1 - 8, 8):
        for j in np.arange(0, N+1 -8 , 8):
            x_block = x[ i : i + 8, j : j + 8 ]
            block_DCT = DCT_2D(x_block)
            X_block = block_DCT.solve2()
            X_block_truncated, index = compress_block(X_block, K)
            X_compressed[i : i + 8, j : j + 8 ] = X_block_truncated
            
    return X_compressed

def compress_image_DFT(x, K):
    """
    Partition the image into 8*8 blocks, compute the DFT of each block and compress
    """ 
    N = x.shape[0]
    
    X_compressed = np.zeros([N, N], dtype=np.complex)
    for i in np.arange(0, N+1 - 8, 8):
        for j in np.arange(0, N+1 -8 , 8):
            x_block = x[ i : i + 8, j : j + 8 ]
            block_DCT = DFT_2D(x_block)
            X_block = block_DCT.solve2()
            X_block_truncated, index = compress_block(X_block, K)
            X_compressed[i : i + 8, j : j + 8 ] = X_block_truncated
            
    return X_compressed
        
            
def q12(img, K):
    """
    Compress the input image with the input parameter K
    """ 
    
    
    X_img_DCT = compress_image_DCT(img, K)
    
    plt.imshow(np.log(1+np.abs(X_img_DCT)*100), cmap='gray')
    plt.title('Compressed DCT of the partitioned image')
    plt.colorbar()
    plt.show() 
    
    X_img_DFT = compress_image_DFT(img, K)
    
    plt.imshow(np.log(1+np.abs(X_img_DFT)*100), cmap='gray')
    plt.title('Compressed DFT of the partitioned image')

    plt.colorbar()
    plt.show() 
    
    return X_img_DCT
    
    
def q14_1(X):
    """
    Reconstruct the image with the compressed DCT matrix
    """ 
    
    N = X.shape[0]

    x_reconstruct = np.zeros([N, N], dtype=np.complex)
    for i in np.arange(0, N+1 - 8, 8):
        for j in np.arange(0, N+1 -8 , 8):
            X_block = X[ i : i + 8, j : j + 8 ]
            block_iDCT = iDCT_2D(X_block)
            x_block = block_iDCT.solve2()
            x_reconstruct[i : i + 8, j : j + 8 ] = x_block
    x_recons_norm = abs(x_reconstruct) / np.amax(abs(x_reconstruct))
            
    plt.imshow(x_recons_norm, cmap='gray')

    plt.colorbar()
    plt.show() 
    
    return abs(x_reconstruct)
    
    
    
    
    
  
###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__': 
    
    img = mpimg.imread('imgB_prenoise.png')  
    
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()
    K_seq = [4,8,16,32]
    rho = np.zeros(len(K_seq))
    
    
    for i in np.arange(len(K_seq)):
        X_DCT = q12(img, K_seq[i])
        x_K = q14_1(X_DCT)
        # compute the reconstructed error
        rho[i] = np.linalg.norm(img - x_K, 2)
        
    plt.plot(K_seq, rho)
    plt.title('Reconstructed errors')
    plt.savefig('recons_error.png')
    plt.xlabel('K')
    plt.ylabel('rho_K')

        
        

    
