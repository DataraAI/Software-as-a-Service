import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def masks_png_mp4(image_folder):
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    #images.sort()  # Ensure frames are in the correct order

    # Read the first image to get dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fps=24
    video_name='BMW_Grille_Mask'

    codec = cv2.VideoWriter.fourcc('')
    video = cv2.VideoWriter(video_name, codec, fps, (width, height))

    # Append images to the video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video writer
    video.release()
    print(f"Video saved as {video_name}")


def masks_npy_to_png(input_dir,output_dir):
    input_path=input_dir + 'mask_' + f"{1:03d}" + '.npy'
    #print(input_path)
    frame=np.load(input_path)
    plt.imshow(frame, cmap='gray')
    plt.show()
    frame[frame != 1] = 0
    #plt.imshow(frame, cmap='gray')
    #plt.imsave(output_dir + 'mask_' + f"{i:03d}" + '.png', frame, cmap='gray')