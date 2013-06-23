# pyssim

This fork differs in two aspects.

Input:
 * Numpy arrays
 * in BGR color mode (from OpenCV)
 * no alpha channel support

Performance:

 * Convolves 1D Gaussian filters, reused
 * uses numexpr
 * around 3-10x speedup, depending on image size
