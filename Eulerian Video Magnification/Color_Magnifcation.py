import cv2
import numpy as np
import tqdm

# Constants
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

# YIQ to RGB conversion kernel
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

# Parameters, adjust your parameters here
level = 4
alpha = 100
low_omega = 1
high_omega = 20
attenuation = 1
freq_range = [low_omega, high_omega]
kernel = gaussian_kernel

# Helper Methods
def loadVideo(video_path):
    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        ret, frame = video.read()

        if ret is False:
            break

        image_sequence.append(frame[:, :, ::-1])

        if cv2.waitKey(1) == ord('q'):
            break
    video.release()

    return np.asarray(image_sequence), fps


def saveVideo(video, saving_path, fps):
    (height, width) = video[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(saving_path, fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(len(video)), ascii=True, desc="Saving Video"):
        writer.write(video[i][:, :, ::-1])

    writer.release()


def rgb2yiq(rgb_image):
    image = rgb_image.astype(np.float32)
    return image @ yiq_from_rgb.T


def yiq2rgb(yiq_image):
    image = yiq_image.astype(np.float32)
    return image @ rgb_from_yiq.T


def pyrDown(image, kernel):
    return cv2.filter2D(image, -1, kernel)[::2, ::2]


def pyrUp(image, kernel, dst_shape=None):
    dst_height = image.shape[0] + 1
    dst_width = image.shape[1] + 1

    if dst_shape is not None:
        dst_height -= (dst_shape[0] % image.shape[0] != 0)
        dst_width -= (dst_shape[1] % image.shape[1] != 0)

    height_indexes = np.arange(1, dst_height)
    width_indexes = np.arange(1, dst_width)

    upsampled_image = np.insert(image, height_indexes, 0, axis=0)
    upsampled_image = np.insert(upsampled_image, width_indexes, 0, axis=1)

    return cv2.filter2D(upsampled_image, -1, 4 * kernel)


def idealTemporalBandpassFilter(images,
                                fps,
                                freq_range,
                                axis=0):

    fft = np.fft.fft(images, axis=axis)
    frequencies = np.fft.fftfreq(images.shape[0], d=1.0/fps)

    low = (np.abs(frequencies - freq_range[0])).argmin()
    high = (np.abs(frequencies - freq_range[1])).argmin()

    fft[:low] = 0
    fft[high:] = 0
    real_part = np.real(fft[:, :, 0])
    magnitude = np.abs(real_part)
    peak_index = np.argmax(magnitude[:, 0, 0])
    peak_frequency = frequencies[peak_index]

    return np.fft.ifft(fft, axis=0).real

# Gaussian Methods
def gaussian_evm(images,
                 fps,
                 kernel,
                 level,
                 alpha,
                 freq_range,
                 attenuation):

    gaussian_pyramids = getGaussianPyramids(
        images=images,
        kernel=kernel,
        level=level
    )

    print("Gaussian Pyramids Filtering...")
    filtered_pyramids = filterGaussianPyramids(
        pyramids=gaussian_pyramids,
        fps=fps,
        freq_range=freq_range,
        alpha=alpha,
        attenuation=attenuation
    )
    print("Finished!")

    output_video = getGaussianOutputVideo(
        original_images=images,
        filtered_images=filtered_pyramids
    )

    return output_video


def generateGaussianPyramid(image, kernel, level):
    image_shape = [image.shape[:2]]
    downsampled_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=downsampled_image, kernel=kernel)
        image_shape.append(downsampled_image.shape[:2])

    gaussian_pyramid = downsampled_image
    for curr_level in range(level):
        gaussian_pyramid = pyrUp(
            image=gaussian_pyramid,
            kernel=kernel,
            dst_shape=image_shape[level - curr_level - 1]
        )

    return gaussian_pyramid


def getGaussianPyramids(images, kernel, level):
    gaussian_pyramids = np.zeros_like(images, dtype=np.float32)

    for i in tqdm.tqdm(range(images.shape[0]),
                       ascii=True,
                       desc='Gaussian Pyramids Generation'):

        gaussian_pyramids[i] = generateGaussianPyramid(
            image=rgb2yiq(images[i]),
            kernel=kernel,
            level=level
        )

    return gaussian_pyramids


def filterGaussianPyramids(pyramids,
                           fps,
                           freq_range,
                           alpha,
                           attenuation):

    filtered_pyramids = idealTemporalBandpassFilter(
        images=pyramids,
        fps=fps,
        freq_range=freq_range
    ).astype(np.float32)

    filtered_pyramids *= alpha
    filtered_pyramids[:, :, :, 1:] *= attenuation

    return filtered_pyramids


def getGaussianOutputVideo(original_images, filtered_images):
    video = np.zeros_like(original_images)

    for i in tqdm.tqdm(range(filtered_images.shape[0]),
                       ascii=True,
                       desc="Video Reconstruction"):

        video[i] = reconstructGaussianImage(
            image=original_images[i],
            pyramid=filtered_images[i]
        )

    return video


def reconstructGaussianImage(image, pyramid):
    reconstructed_image = rgb2yiq(image) + pyramid
    reconstructed_image = yiq2rgb(reconstructed_image)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    return reconstructed_image.astype(np.uint8)


# Main
images, fps = loadVideo('sunset.mp4')
output_video = gaussian_evm(
    images, fps, kernel, level, alpha, freq_range, attenuation)
saveVideo(video=output_video, saving_path='sunsetamplified.mp4', fps=fps)
