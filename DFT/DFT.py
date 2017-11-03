# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy as np
import math

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        r=np.shape(matrix)[0]
        c=np.shape(matrix)[1]
        #print(r,c)
        N=r
        final = np.zeros((r, c), dtype=np.complex_)
        u=np.shape(final)[0]
        v=np.shape(final)[1]
        for u in range(0,N):
            for v in range(0,N):
                for i in range(0,N):
                    for j in range(0,N):
                        final[u][v]=matrix[i][j]*(math.cos(2*math.pi*(u*i+v*j)/N)-1j*math.sin(2*math.pi*(u*i+v*j)/N))+final[u][v]



        #print(np.fft.fft2(matrix))



        return final

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        r=np.shape(matrix)[0]
        c=np.shape(matrix)[1]
        #print(r,c)
        N=r
        finalift = np.zeros((r, c), dtype=np.complex_)
        u=np.shape(finalift)[0]
        v=np.shape(finalift)[1]

        for i in range(0,N):
            for j in range(0,N):
                for u in range(0,N):
                    for v in range(0,N):
                        finalift[i][j]+=matrix[u][v]*(math.cos(2*math.pi*(u*i+v*j)/N)+1j*math.sin(2*math.pi*(u*i+v*j)/N))


        #print("Inbuilkt")
        #print(np.fft.ifft2(matrix))


        return finalift


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        r=np.shape(matrix)[0]
        c=np.shape(matrix)[1]
        #print(r,c)
        N=r
        final = np.zeros((r, c))
        u=np.shape(final)[0]
        v=np.shape(final)[1]
        for u in range(0,N):
            for v in range(0,N):
                for i in range(0,N):
                    for j in range(0,N):
                        final[u][v]=matrix[i][j]*(math.cos(2*math.pi*(u*i+v*j)/N))+final[u][v]



        #print(final)


        return final


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        r=np.shape(matrix)[0]
        c=np.shape(matrix)[1]
        #print(r,c)
        N=r
        final = np.zeros((r, c))
        for u in range(0,N):
            for v in range(0,N):
                final[u][v]=math.sqrt(matrix[u][v].real*matrix[u][v].real + matrix[u][v].imag*matrix[u][v].imag)





        return final