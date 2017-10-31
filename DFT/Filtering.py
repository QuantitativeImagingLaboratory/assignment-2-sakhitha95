# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import numpy as np
import cv2
import math
class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        # for i in range(0,shape):
        #     for j in range(0,shape):
        #         mask[i][j]=
        p=shape[0]
        q=shape[1]
        mask=np.zeros((p,q))
        for u in range(p):
            for v in range(q):
                d=math.sqrt((u-(p-1)/2)*(u-(p-1)/2)+(v-(q-1)/2)*(v-(q-1)/2))
                if d<=cutoff:
                    mask[u][v]=1
                else:
                    mask[u][v]=0

        # print(mask)
        # cv2.imshow("image",mask)
        # cv2.waitKey(0)
        return mask


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        p = shape[0]
        q = shape[1]
        mask=np.zeros((p,q))
        mask=1-self.get_ideal_low_pass_filter(shape, cutoff)


        # cv2.imshow("image", mask)
        # cv2.waitKey(0)
        #
        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        order=self.order
        p = shape[0]
        q = shape[1]
        mask = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                d = math.sqrt((u - (p - 1) / 2) * (u - (p - 1) / 2) + (v - (q - 1) / 2) * (v - (q - 1) / 2))
                h = 1 / (1 + ((d / cutoff) ** (2 * order)))

                mask[u][v] = h

        # cv2.imshow("Image", mask)
        # cv2.waitKey(0)
        
        return mask

    def get_butterworth_high_pass_filter(self, shape, cutoff):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        p = shape[0]
        q = shape[1]
        mask = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                d = math.sqrt((u - (p - 1) / 2) * (u - (p - 1) / 2) + (v - (q - 1) / 2) * (v - (q - 1) / 2))
                h = 1 / (1 + (cutoff/d )** (2 * self.order))

                mask[u][v] = h

        # cv2.imshow("Image", mask)
        # cv2.waitKey(0)
        return mask

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        p = shape[0]
        q = shape[1]
        mask = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                d = math.sqrt((u - (p - 1) / 2) * (u - (p - 1) / 2) + (v - (q - 1) / 2) * (v - (q - 1) / 2))
                h = math.exp(-(d * d) / (2 * cutoff * cutoff))

                mask[u][v] = h

        # cv2.imshow("Image", mask)
        # cv2.waitKey(0)
        
        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        p = shape[0]
        q = shape[1]
        mask = np.zeros((p, q))
        for u in range(p):
            for v in range(q):
                d = math.sqrt((u - (p - 1) / 2) * (u - (p - 1) / 2) + (v - (q - 1) / 2) * (v - (q - 1) / 2))
                h = math.exp(-(d * d) / (2 * cutoff * cutoff))

                mask[u][v] = 1-h

        cv2.imshow("Image", mask)
        cv2.waitKey(0)
        
        return mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        a = 0
        b = 255
        c = np.min(image)
        d = np.max(image)
        print(c, d)
        display = np.zeros((np.shape(image)[0], np.shape(image)[1]), dtype="uint8")
        for i in range(0, np.shape(image)[0]):
            for j in range(0, np.shape(image)[1]):
                display[i][j] = (image[i][j] - c) * ((b - a) / (d - c)) + a

        return display

    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8
        """
        f = np.fft.fft2(self.image)
        s = np.fft.fftshift(f)
        # comp=np.log(np.absolute(s))*6
        # cmp1=np.uint8(comp)
        # cv2.imshow("image",cmp1)
        # cv2.waitKey(0)
        # magnitude of dft
        dftmag = np.zeros((np.shape(s)[0], np.shape(s)[1]))
        for i in range(0, np.shape(s)[0]):
            for j in range(0, np.shape(s)[1]):
                dftmag[i][j] = math.sqrt(s[i][j].real * s[i][j].real + s[i][j].imag * s[i][j].imag)
        dftfinalmag = np.uint8(np.log(dftmag) * 12)
        sh = []
        sh.append(np.shape(s)[0])
        sh.append(np.shape(s)[1])
        print(sh)

        m = self.filter(sh, self.cutoff)

        final = s * m
        # comp = np.log(np.absolute(final)) * 6
        # cmp1 = np.uint8(comp)
        # cv2.imshow("image", cmp1)
        # cv2.waitKey(0)
        magfinal = np.zeros((np.shape(final)[0], np.shape(final)[1]))
        for i in range(0, np.shape(final)[0]):
            for j in range(0, np.shape(final)[1]):
                magfinal[i][j] = math.sqrt(final[i][j].real * final[i][j].real + final[i][j].imag * final[i][j].imag)
        filtermag = self.post_process_image(np.uint8(np.log(magfinal) * 12))
        invs = np.fft.ifftshift(final)
        infft = np.fft.ifft2(invs)
        mag = np.zeros((np.shape(infft)[0], np.shape(infft)[1]))
        for i in range(0, np.shape(infft)[0]):
            for j in range(0, np.shape(infft)[1]):
                mag[i][j] = math.sqrt(infft[i][j].real * infft[i][j].real + infft[i][j].imag * infft[i][j].imag)

        cv2.imshow("image", mag)
        cv2.waitKey(0)
        # mag = np.log(mag) * 12

        final_filt = self.post_process_image(mag)

        # cv2.imshow("image", final_filt)
        # cv2.waitKey(0)
        return [final_filt, dftfinalmag, filtermag]
