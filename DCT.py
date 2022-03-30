# MELANIE HERBERT
# ESE 224
# LAB 9

import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from scipy.fft import dct, idct

import discrete_signal

# QUESTION 1.1
# IN ORDER TO COMPUTE THE DCT WE CAN TAKE ADVANTAGE OF THE RELATIONSHIP WITH INNER PRODUCTS
# DRAWING ON OUR KNOWLEDGE FROM THE LAST LAB, WE CAN SWITCH THE COMPLEX EXPONENTIAL MATRIX
# NOW FOR THE DISCRETE COSINE MATRIX
class inner_prod_2D():
    def __init__(self, x, y):
       # INPUTS X AND Y, BOTH TWO-DIMENSIONAL SIGNALS
        self.x=x
        self.y=y
        self.N=np.shape(x)[0]

# METHOD COMPUTES THE INNER PRODUCT
    def solve(self):
        prod = 0        
        for i in range(self.N):
            for j in range(self.N):
                #BETWEEN TWO DIMENSIONAL SIGNALS X AND Y
                prod = prod + self.x[i,j] * np.conj(self.y[i,j])
            
        return prod

# IN QUESTION 1.1 WE NEED TO REPLACE THE DISCRETE COSINE MATRIX PREVIOUSLY USED IN THE INNER PRODUCT
# EQUATION INSTEAD WITH A DISCRETE COSINE MATRIX.

# PYTHON CLASS CREATES DISCRETE COSINE MATRIX
# FREQUENCY: K AND L, DISCRETE
# DURATION: M AND N OF COSINE MATRIX

class Generate_Discrete_Cosine_2D(object):
    def __init__(self, k, l, M, N):
        self.k = k
        self.l = l
        self.M = M
        self.N = N

        # VECTOR WITH ELEMENTS OF TIME INDEXES
        self.n = np.arange(N)
        self.m = np.arange(M)

        # VECTOR WITH ELEMENTS OF COMPLEX EXPONENTIAL
        self.dis_cos_2d = np.matmul( np.cos(self.k * cmath.pi / 2/self.M * (2*self.m+1) ).reshape((self.M,1)) ,np.cos(self.l * cmath.pi / 2/self.N * (2*self.n+1)).reshape((1,self.N)) )


# QUESTION 1.1 DCT IN TWO DIMENSIONS
# ABOVE CODE AQUIRES THE PIECES NEEDED TO COMPUTE THE DCT IN 2-D
# GOAL: INPUT 2-D SIGNAL OF SIZE N SQUARED
# OUTPUT: 2D DCT

# INPUTS TIME DOMAIN SIGNAL X
class DCT_2D():
    def __init__(self, x):
        self.x = x
        self.M = np.shape(x)[0]
        self.N = np.shape(x)[1]

# COMPUTE DCT OF X
    def solve(self):
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

                # THE DCT CAN BE REPRESENTED AS THE INNER PRODUCT FOR A SIGNAL WITH DISCRETE COSINE
                # WE CAN CALL ON THE INNER_PROD_2D PYTHON CLASS USED IN THE PREVIOUS LAB AND JUST SWAP OUT
                # THE COMPLEX EXPONENTIAL MATRIX WITH THE DISCRETE COSINE MATRIX
                twod_cos = Generate_Discrete_Cosine_2D(k,l,self.M, self.N).dis_cos_2d
                inner_prod = inner_prod_2D(self.x, twod_cos)
                X[k, l] =  2 / np.sqrt(self.M)/ np.sqrt(self.N) *c1 *c2 * inner_prod.solve()
        return X

    # ALTERNATIVE METHOD OF COMPUTING THE DCT OF SIGNAL X BY USING THE BUILD-IN DCT FUNCTION.
    # COMES IN USE FOR WHEN THE SIGNAL DIMENSION IS VERY LARGE AND IT TAKES A LONG TIME TO COMPUTE THE NESTED
    # FOR LOOP STRUCTURE
    def solve2(self):
        return dct(dct(self.x.T, norm='ortho').T, norm='ortho')
    
###################################################################################################
################################################################################################
################################################################################################

## COMPUTES IDCT OF X USING BUILT IN FUNCTION
class iDCT_2D():
    def __init__(self, X):
        self.X = X
    
    def solve2(self):
        return idct(idct(self.X.T, norm='ortho').T, norm='ortho')

class DFT_2D():
    def __init__(self, x):
        self.x=x
        self.M=np.shape(x)[0]
        self.N=np.shape(x)[1]

    # WE CAN CALL ON THE INNER_PROD_2D PYTHON CLASS USED IN THE PREVIOUS LAB AND JUST SWAP OUT
    # THE COMPLEX EXPONENTIAL MATRIX WITH THE DISCRETE COSINE MATRIX
    def solve(self):
        X=np.zeros([self.M, self.N], dtype=np.complex)
        for m in range(self.M):
            for n in range(self.N):
                for i in range(self.M):
                    for j in range(self.N):
                        X[m,n] = X[m,n] + self.x[i,j]/np.sqrt(self.M*self.N)*np.exp(-1j*2*cmath.pi*(m*i/self.M+n*j/self.N))
            
        return X

    # ALTERNATIVE METHOD OF COMPUTING THE DFT OF SIGNAL X BY USING THE BUILD-IN DFT FUNCTION.
    # COMES IN USE FOR WHEN THE SIGNAL DIMENSION IS VERY LARGE AND IT TAKES A LONG TIME TO COMPUTE THE NESTED
    # FOR LOOP STRUCTURE
    def solve2(self):
        X = np.fft.fft2(self.x) / np.sqrt(self.M*self.N)
        X_shift = np.fft.fftshift(X)

        return X_shift

##### QUESTION 1.3 QUANTIZATION
# Write a function that executes the above procedure (implementing a basic version of the JPEG compression scheme).
# If your code is running too slowly, try using Python's built-in functions.
# 1)Extract an 8x8 block of the image. Recall that our image signal corresponds to an  matrix. That is what we would obtain after importing an image.
# Creating the blocks or patches, we can think of each block as a ’submatrix’
# 2) Compute the DCT of each block. Then store that resulting signal in X.
# 3) Now we should have access to the frequency components of the signal. Then we quantize the DCT coefficients (given the equation in the packet).

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
    

#### QUESTION 1.2 PART A
def compress_image_DCT(x, K):
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

#### QUESTION 1.2 PART A
## TAKES IN SIGNAL (IMAGE) OF SIZE N SQUARED AND PARTITIONS INTO PATCHES OF SIZE 8X8
# STORES THE K LARGEST DFT COEFFICIENTS AND ASSOCIATED FREQUENCIES
def compress_image_DFT(x, K):

    # COMPUTES THE DFT OF EACH BLOCK AND COMPRESSES
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
        
### QUESTION 1.2 PART B- IMPLEMENTATING PART A AND PART B
## DISPLAYS PLOTS
def image_compression_plot(img, K):
    # CALLING THE COMPRESSION OF INPUT IMAGE
    # INPUT FREQUENCY K
    X_img_DCT = compress_image_DCT(img, K)

    # CREATE IMAGE COMPRESSION PLOTS DCT
    plt.imshow(np.log(1+np.abs(X_img_DCT)*100), cmap='gray')
    plt.title('Compressed DCT of the partitioned image')
    plt.colorbar()
    plt.show()

    # CREATE IMAGE COMPRESSION PLOTS DFT
    X_img_DFT = compress_image_DFT(img, K)
    
    plt.imshow(np.log(1+np.abs(X_img_DFT)*100), cmap='gray')
    plt.title('Compressed DFT of the partitioned image')

    plt.colorbar()
    plt.show() 
    
    return X_img_DCT
    

#### QUESTION 1.4

## USING THE COMPRESSED DCT MATRIX
"""
DCT compression, also known as block compression, compresses data in sets of discrete DCT blocks. 
DCT blocks can have a number of sizes, including 8x8 pixels for the standard DCT, 
and varied integer DCT sizes between 4x4 and 32x32 pixels.

"""

"""

The DCT can be used to convert the signal (spatial information) into numeric data ("frequency" or "spectral" information)
so that the image's information exists in a quantitative form that can be manipulated for compression. The signal for a 
graphical image can be thought of as a three-dimensional signal.

"""
def question_image_reconstruction_1_4(X):
    
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

if __name__ == '__main__': 
    
    img = mpimg.imread('imgB_prenoise.png')  
    
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()
    K_seq = [4,8,16,32]
    rho = np.zeros(len(K_seq))
    
    for i in np.arange(len(K_seq)):
        X_DCT = image_compression_plot(img, K_seq[i])
        x_K = question_image_reconstruction_1_4(X_DCT)
        rho[i] = np.linalg.norm(img - x_K, 2)

    # DISPLAY QUANTIZED PLOTS
    plt.plot(K_seq, rho)
    plt.title('Reconstructed errors')
    plt.savefig('recons_error.png')
    plt.xlabel('K')
    plt.ylabel('rho_K')

        
        

    
