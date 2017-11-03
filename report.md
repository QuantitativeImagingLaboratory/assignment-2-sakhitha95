
Forward fourier transformation:
?	Using forward fourier transformation we decompose an image to sine and cosine components.
?	The output of the transformation represents the image in the Fourier or frequency domain.
?	The input image is the spatial domain equivalent.
?	In the Fourier domain image, each point represents a particular frequency contained in the spatial domain image.

?	where f(a,b) is the image in the spatial domain and the exponential term is the basis function corresponding to each point F(k,l) in the Fourier space.
?	The shape of the original matrix is taken and the output image is defined.
?	Then iterated from left to right and top to bottom. Calculate the fft value at each point.
?	The Fourier Transform produces a complex number
Inverse Fourier Transformation:
?	Fourier image can be re-transformed to the spatial domain.  The inverse Fourier transform is given by:

?	The shape of the original matrix is taken and the output image is defined.
?	Then iterated from left to right and top to bottom. Calculate the inverse fft value at each point.
?	The Fourier Transform produces a complex number

Discrete cosine transform:
?	It is the real part of the output image obtained after fourier transformation that is we decompose the image only to the cosine components.

Magnitude:
?	In image processing, often only the magnitude of the Fourier Transform is displayed, as it contains most of the information of the geometric structure of the spatial domain image. It is the absolute value of the output image at each pixel.

Masks:

ideal low pass:
?	Keep frequencies below a certain frequency. A Low pass mask with original image size is created. Which is used in convolution. It blurs image
?	We calculate the distance from the center if it is less than the cutoff we assign it 1 that is white color. Else black color.
?	Ringing Effect is observed. When the cutoff increases the image appears to be more smoother.

Ideal High Pass Filter:
?	Opposite of low pass filtering. eliminate low frequency values keeping others. High pass filtering causes image sharpening.
?	The mask is created by doing one minus the lowpass filter. Ringing Effect is observed.

Lowpass Butterworth Filtering:
?	It blurs image. Sharp cutoffs in ideal pass filter causes ringing effect. To avoid ringing, can use circle with more gentle cutoff slope.
?	Butterworth filters have more gentle cutoff slopes. Butterworth Lowpass Filters (BLPF) of order and with cutoff frequency
?	We calculated the value at certain pixel using the below formula. Where D is the distance from the center and D0 is the cutoff.

Highpass Butterworth Filtering:
?	Opposite of low pass filtering. eliminate low frequency values keeping others. High pass filtering causes image sharpening.
?	The mask is created by doing one minus the lowpass filter.

Gaussian Low Pass:
?	Gaussian filtering, the frequency coefficients are not cut abruptly, but smoother cut off process is used instead.
?	We calculated the value at certain pixel using the below formula. Where D is the distance from the center and ? is the cutoff.

Gaussian High Pass:
?	Opposite of low pass filtering. eliminate low frequency values keeping others. High pass filtering causes image sharpening.
?	The mask is created by doing one minus the lowpass filter.

Filtering:
?	The fft of the image is calculated
?	we shift the fft to center the low frequencies
?	The required mask is calculated
?	The image is filtered based on the mask (Convolution theorem)
?	inverse shift of the filtered image is computed
?	inverse fourier transform is calculated after inverse shift is done.
?	Magnitude is computed after the inverse fourier transform.
?	Full contrast stretch is done and the final image is filtered.



