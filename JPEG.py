# MELANIE HERBERT
# ESE 224 LAB 9

# QUESTION 1.3

import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from scipy.fft import dct, idct


# BELOW CODE AQUIRES THE PIECES NEEDED TO COMPUTE THE DCT IN 2-D
# GOAL: INPUT 2-D SIGNAL OF SIZE N SQUARED
# OUTPUT: 2D DCT

# INPUTS TIME DOMAIN SIGNAL X
class DCT_2D():
    def __init__(self, x):
        self.x = x
        self.M = np.shape(x)[0]
        self.N = np.shape(x)[1]

    # ALTERNATIVE METHOD OF COMPUTING THE DCT OF SIGNAL X BY USING THE BUILD-IN DCT FUNCTION.
    # COMES IN USE FOR WHEN THE SIGNAL DIMENSION IS VERY LARGE AND IT TAKES A LONG TIME TO COMPUTE THE NESTED
    # FOR LOOP STRUCTURE
    def solve2(self):
        return dct(dct(self.x.T, norm='ortho').T, norm='ortho')
    
# QUESTION 1.4
class iDCT_2D():
    def __init__(self, X):
        self.X = X
    
    def solve2(self): # COMPUTES THE IDCT OF X USING THE BUILT IN FUNCTION
        return idct(idct(self.X.T, norm='ortho').T, norm='ortho')


## TABLE GIVEN FROM QUESTION 1.3

##### QUESTION 1.3 QUANTIZATION
# Write a function that executes the above procedure (implementing a basic version of the JPEG compression scheme).
# If your code is running too slowly, try using Python's built-in functions.
# 1)Extract an 8x8 block of the image. Recall that our image signal corresponds to an  matrix. That is what we would obtain after importing an image.
# Creating the blocks or patches, we can think of each block as a ’submatrix’
# 2) Compute the DCT of each block. Then store that resulting signal in X.
# 3) Now we should have access to the frequency components of the signal. Then we quantize the DCT coefficients (given the equation in the packet).
def quantization_table(quality):
    matrix_Q_L = [ [16, 11, 10, 16, 24, 40, 51, 61],
           [12, 12, 14, 19, 26, 58, 60, 55],
    [ 14, 13, 16, 24, 40, 57, 69, 56],
    [ 14, 17, 22, 29, 51, 87, 80, 62],
    [ 18, 22, 37, 56, 68, 109, 103, 77],
    [ 24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]]
    
    block_matrix = np.zeros((8, 8))
    if quality > 50:
        block_matrix =( matrix_Q_L * (np.ones((8,8)) * ( ( 100 - quality ) / 50 ) )).round().astype(np.int32)
    else:
        block_matrix = (matrix_Q_L *( np.ones((8,8)) * ( 50 / quality))).round().astype(np.int32)
         
    return block_matrix


def quantize(block_matrix, block):
       
    return (block /block_matrix).round().astype(np.int32)

def dequantize(block_matrix, block):
         
    return (( block) * block_matrix).astype(np.int32)


##### QUESTION 1.3 QUANTIZATION

# 1)Extract an 8x8 block of the image. Recall that our image signal corresponds to an  matrix. That is what we would obtain after importing an image.
# Creating the blocks or patches, we can think of each block as a ’submatrix’
# 2) Compute the DCT of each block. Then store that resulting signal in X.
# 3) Now we should have access to the frequency components of the signal. Then we quantize the DCT coefficients (given the equation in the packet).
def JPEG_question_1_3(quantization):
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

###########################
### QUESTION 1.4 PART 2

def quantization_question_1_4_partb(quantization, X_quan):
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

if __name__ == '__main__':
    img = mpimg.imread('imgB_prenoise.png')  

    # QUANTIZED PLOTS
    quan_tab = quantization_table(50)            
    X = JPEG_question_1_3(quan_tab)
    x = def quantization_question_1_4_partb(quantization, X_quan):(quan_tab, X)
    plt.imshow(abs(x), cmap='gray')
    plt.title('Reconstructed image with standard quantization matrix')
    plt.savefig('Recons_jpeg.png')
    print(np.linalg.norm(img - abs(x), 2))
    plt.colorbar()
    plt.show()    

"""
Looking at the reconstructed image with varying values for the quantization matrix (see above three images, 
one with a high value quantization matrix, lower value, and then a standard value). We can see that the quantization 
matrix with a lower value correlates to improved reconstruction rather then the standard quantization matrix. 
Reconstruction will become worse as it eliminates more frequency components and this is shown with the higher value 
for a quantization matrix giving worse reconstruction.

"""
    # 1.3 TO COMPRESS AND RECONSTRUCT WITH A LOWER VALUE QUANTIZATION MATRIX
    quan_tab = quantization_table(95)      
    print(quan_tab)      
    X = JPEG_question_1_3(quan_tab)
    x = def quantization_question_1_4_partb(quantization, X_quan):(quan_tab, X)
    plt.imshow(abs(x), cmap='gray')
    print(np.linalg.norm(img - abs(x), 2))

    plt.title('Reconstructed image with LOWER value quantization matrix')
    plt.savefig('Reconslow_jpeg.png')

    plt.colorbar()
    plt.show()  

    quan_tab = quantization_table(2)   
    print(quan_tab)      

    # Reconstruction will become worse as it eliminates more frequency components and this is shown with the higher value
    # for a quantization matrix giving worse reconstruction.

    # 1.4 TO COMPRESS AND RECONSTRUCT WITH A HIGHER VALUE QUANTIZATION MATRIX
    X = JPEG_question_1_3(quan_tab)
    x = def quantization_question_1_4_partb(quantization, X_quan):(quan_tab, X)
    plt.imshow(abs(x), cmap='gray')
    print(np.linalg.norm(img - abs(x), 2))

    plt.title('Reconstructed image with HIGHER value quantization matrix')
    plt.savefig('Reconshigh_jpeg.png')

    plt.colorbar()
    plt.show()  


   



