import cv2
import numpy as np
import tqdm
from scipy.signal import butter


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


# Parameters
level = 7
alpha = 100
lambda_cutoff = 1000
low_omega = 0.833
high_omega = 1
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


def generateLaplacianPyramid(image, kernel, level):
    laplacian_pyramid = []
    prev_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=prev_image, kernel=kernel)
        upsampled_image = pyrUp(image=downsampled_image,
                                kernel=kernel,
                                dst_shape=prev_image.shape[:2])
        laplacian_pyramid.append(prev_image - upsampled_image)
        prev_image = downsampled_image

    return laplacian_pyramid


def getLaplacianPyramids(images, kernel, level):
    laplacian_pyramids = []

    for image in tqdm.tqdm(images,
                           ascii=True,
                           desc="Laplacian Pyramids Generation"):

        laplacian_pyramid = generateLaplacianPyramid(
            image=rgb2yiq(image),
            kernel=kernel,
            level=level
        )
        laplacian_pyramids.append(laplacian_pyramid)

    return np.asarray(laplacian_pyramids, dtype='object')


def filterLaplacianPyramids(pyramids,
                            level,
                            fps,
                            freq_range,
                            alpha,
                            lambda_cutoff,
                            attenuation):

    filtered_pyramids = np.zeros_like(pyramids)
    delta = lambda_cutoff / (8 * (1 + alpha))
    b_low, a_low = butter(1, freq_range[0], btype='low', output='ba', fs=fps)
    b_high, a_high = butter(1, freq_range[1], btype='low', output='ba', fs=fps)

    lowpass = pyramids[0]
    highpass = pyramids[0]
    filtered_pyramids[0] = pyramids[0]

    for i in tqdm.tqdm(range(1, pyramids.shape[0]),
                       ascii=True,
                       desc="Laplacian Pyramids Filtering"):

        lowpass = (-a_low[1] * lowpass
                   + b_low[0] * pyramids[i]
                   + b_low[1] * pyramids[i - 1]) / a_low[0]
        highpass = (-a_high[1] * highpass
                    + b_high[0] * pyramids[i]
                    + b_high[1] * pyramids[i - 1]) / a_high[0]

        filtered_pyramids[i] = highpass - lowpass

        for lvl in range(1, level - 1):
            (height, width, _) = filtered_pyramids[i, lvl].shape
            lambd = ((height ** 2) + (width ** 2)) ** 0.5
            new_alpha = (lambd / (8 * delta)) - 1

            filtered_pyramids[i, lvl] *= min(alpha, new_alpha)
            filtered_pyramids[i, lvl][:, :, 1:] *= attenuation

    return filtered_pyramids


def getLaplacianOutputVideo(original_images, filtered_images, kernel):
    video = np.zeros_like(original_images)

    for i in tqdm.tqdm(range(original_images.shape[0]),
                       ascii=True,
                       desc="Video Reconstruction"):

        video[i] = reconstructLaplacianImage(
            image=original_images[i],
            pyramid=filtered_images[i],
            kernel=kernel
        )

    return video


def reconstructLaplacianImage(image, pyramid, kernel):
    reconstructed_image = rgb2yiq(image)

    for level in range(1, pyramid.shape[0] - 1):
        tmp = pyramid[level]
        for curr_level in range(level):
            tmp = pyrUp(tmp, kernel, pyramid[level - curr_level - 1].shape[:2])
        reconstructed_image += tmp.astype(np.float32)

    reconstructed_image = yiq2rgb(reconstructed_image)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)

    return reconstructed_image.astype(np.uint8)


def laplacian_evm(images,
                  fps,
                  kernel,
                  level,
                  alpha,
                  lambda_cutoff,
                  freq_range,
                  attenuation):

    laplacian_pyramids = getLaplacianPyramids(
        images=images,
        kernel=kernel,
        level=level
    )

    filtered_pyramids = filterLaplacianPyramids(
        pyramids=laplacian_pyramids,
        fps=fps,
        freq_range=freq_range,
        alpha=alpha,
        attenuation=attenuation,
        lambda_cutoff=lambda_cutoff,
        level=level
    )

    output_video = getLaplacianOutputVideo(
        original_images=images,
        filtered_images=filtered_pyramids,
        kernel=kernel
    )

    return output_video


def saveVideo(video, saving_path, fps):
    (height, width) = video[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(saving_path, fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(len(video)), ascii=True, desc="Saving Video"):
        writer.write(video[i][:, :, ::-1])

    writer.release()


# Main
images, fps = loadVideo('crane.mp4')
output_video = laplacian_evm(
    images, fps, kernel, level, alpha, lambda_cutoff, freq_range, attenuation)
saveVideo(video=output_video, saving_path='cute.mp4', fps=fps)
