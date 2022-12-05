import numpy as np

'''
Smoothing is a process by which data points are averaged with their neighbours in a series, such as a time series, or image.
This is usually has the effect of blurring the sharp edges in the smoothed data. 
Smoothing is sometimes referred to as filtering, because smoothing has the effect of suppressing high frequency signal and enhancing low frequency signal.
The 'kernel' for smoothing, defines the shape of the function that is used to take the average of the neighboring points. 
A Gaussian kernel is kernel with the shape of a Gaussian(normal distribution) curve. Here is a standard Gaussian, with a mean of 0 and a std of 1
'''

gaussian_kernel = (
    np.array(
        [
            [1, 4,  6,  4,  1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4,  6,  4,  1]

        ]
    )
    / 256
)

yiq_from_rgb = (
    np.array(
        [
            [0.29900000,  0.58700000,  0.11400000],
            [0.59590059, -0.27455667, -0.32134392],
            [0.21153661, -0.52273617,  0.31119955]
        ]

    )
).astype(np.float32)

rgb_from_yiq = np.linalg.inv(yiq_from_rgb)
