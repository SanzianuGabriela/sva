import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os, urllib
import time


def denoise_gaussian(image):


    # aplicăm filtrul Gaussian
    denoised_gaussian = cv2.GaussianBlur(image, (5, 5), 0)

    return denoised_gaussian


def denoise_median_arith(image, kernel_size=3):
    # aplicăm filtrul de medie aritmetică
    image = np.float32(image)
    denoised_median_arith = cv2.medianBlur(image, kernel_size)

    return denoised_median_arith


def denoise_geometric(image, kernel_size=5):
    image = np.float32(image)

    # aplicăm filtrul de medie geometrică
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    denoised_geometric = cv2.filter2D(image, -1, kernel)

    return denoised_geometric



def denoise_median(image, kernel_size=3):
    image = np.float32(image)

    # aplicăm filtrul median
    denoised_median = cv2.medianBlur(image, kernel_size)

    return denoised_median


#@st.cache
def get_model():
    dncnn = tf.keras.models.load_model("dncnn.h5")
    return dncnn


def get_list_of_images():
    file_list = os.listdir(os.path.join(os.getcwd(), 'images'))
    return [str(filename) for filename in file_list if str(filename).endswith('.jpg')]


def models():
    st.title('Atenuarea zgomotului')

    st.write('\n')

    choice = st.sidebar.selectbox("Alege cum să încarci imaginea", ["Din setul de imagini existent", "Din calculatorul personal"])

    if choice == "Din calculatorul personal":
        uploaded_file = st.sidebar.file_uploader("Choose a image file", type="jpg")

        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            gt = cv2.imdecode(file_bytes, 1)
            prediction_ui(gt)

    if choice == "Din setul de imagini existent":

        image_file_chosen = st.sidebar.selectbox('Setul de poze:', get_list_of_images(), 10)

        if image_file_chosen:
            imagespath = os.path.join(os.getcwd(), 'images')
            gt = cv2.imread(os.path.join(imagespath, image_file_chosen))
            prediction_ui(gt)


def PSNR(gt, image, max_value=1):
    """"Function to calculate peak signal-to-noise ratio (PSNR) between two images."""
    height, width, channels = gt.shape
    gt = cv2.resize(gt, (width // 40 * 40, height // 40 * 40), interpolation=cv2.INTER_CUBIC)
    mse = np.mean((gt - image) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def create_image_from_patches(patches, image_shape):
    '''This function takes the patches of images and reconstructs the image'''
    image = np.zeros(image_shape)  # Create a image with all zeros with desired image shape
    patch_size = patches.shape[1]
    p = 0
    for i in range(0, image.shape[0] - patch_size + 1, int(patch_size / 1)):
        for j in range(0, image.shape[1] - patch_size + 1, int(patch_size / 1)):
            image[i:i + patch_size, j:j + patch_size] = patches[p]  # Assigning values of pixels from patches to image
            p += 1
    return np.array(image)


def get_patches(image):
    '''This functions creates and return patches of given image with a specified patch_size'''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    crop_sizes = [1]
    patch_size = 40
    patches = []
    for crop_size in crop_sizes:  # We will crop the image to different sizes
        crop_h, crop_w = int(height * crop_size), int(width * crop_size)
        image_scaled = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h - patch_size + 1, int(patch_size / 1)):
            for j in range(0, crop_w - patch_size + 1, int(patch_size / 1)):
                x = image_scaled[i:i + patch_size,
                    j:j + patch_size]  # This gets the patch from the original image with size patch_size x patch_size
                patches.append(x)
    return patches


def get_image(gt, noise_level):
    patches = get_patches(gt)
    height, width, channels = gt.shape
    test_image = cv2.resize(gt, (width // 40 * 40, height // 40 * 40), interpolation=cv2.INTER_CUBIC)
    patches = np.array(patches)
    ground_truth = create_image_from_patches(patches, test_image.shape)

    # predicting the output on the patches of test image
    patches = patches.astype('float32') / 255.
    patches_noisy = patches + tf.random.normal(shape=patches.shape, mean=0, stddev=noise_level / 255)
    patches_noisy = tf.clip_by_value(patches_noisy, clip_value_min=0., clip_value_max=1.)
    noisy_image = create_image_from_patches(patches_noisy, test_image.shape)

    return ground_truth / 255., noisy_image, patches_noisy


def predict_fun(model, patches_noisy, gt):
    height, width, channels = gt.shape
    gt = cv2.resize(gt, (width // 40 * 40, height // 40 * 40), interpolation=cv2.INTER_CUBIC)
    denoised_patches = model.predict(patches_noisy)
    denoised_patches = tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

    # Creating entire denoised image from denoised patches
    denoised_image = create_image_from_patches(denoised_patches, gt.shape)

    return denoised_image


def prediction_ui(gt):
    dncnn = get_model()
    noise_level = st.sidebar.slider("Alegeți nivelul de zgomot", 0, 45, 0)

    ground_truth, noisy_image, patches_noisy = get_image(gt, noise_level=noise_level)
    st.header('Imaginea Originală')
    st.markdown('** Nivelul de zgomot : ** `%d`  ( Nivelul 0 va reprezenta imaginea originală )' % (noise_level))
    st.image(noisy_image)
    if noise_level != 0:
        st.success('PSNR-ul imaginii cu zgomot : %.3f db' % PSNR(ground_truth, noisy_image))

    model = st.sidebar.radio("Alege un algoritm",
                             ('Gaussian', 'Aritmetic', 'Median', 'Geometric', 'DNCNN'),
                             0)

    submit = st.sidebar.button('Afișează rezultatul')

    if submit and noise_level >= 10:

        if model == "DNCNN":
            progress_bar = st.progress(0)
            start = time.time()
            progress_bar.progress(10)
            denoised_image = predict_fun(dncnn, patches_noisy, gt)
            progress_bar.progress(40)
            end = time.time()
            st.header('Imagine după folosirea modelului DNCNN')

            st.image(denoised_image)
            st.success('PSNR-ul imaginii: %.3f db  ' % (PSNR(ground_truth, denoised_image)))

            progress_bar.progress(100)
            progress_bar.empty()

        if model == 'Gaussian':
            progress_bar = st.progress(0)
            start = time.time()
            progress_bar.progress(10)
            denoised_image = denoise_gaussian(noisy_image)
            progress_bar.progress(40)
            end = time.time()
            st.header('Imagine după aplicarea filtrului Gaussian')

            st.image(denoised_image)
            st.success('PSNR-ul imaginii: %.3f db  ' % (PSNR(ground_truth, denoised_image)))

            progress_bar.progress(100)
            progress_bar.empty()

        elif model == 'Aritmetic':

            progress_bar = st.progress(0)
            start = time.time()
            progress_bar.progress(10)
            denoised_image = denoise_median_arith(noisy_image)
            progress_bar.progress(40)
            end = time.time()
            st.header('Imagine după aplicarea filtrului Aritmetic')

            st.image(denoised_image)
            st.success('PSNR-ul imaginii: %.3f db  ' % (PSNR(ground_truth, denoised_image)))

            progress_bar.progress(100)
            progress_bar.empty()

        elif model == 'Median':
            progress_bar = st.progress(0)
            start = time.time()
            progress_bar.progress(10)
            denoised_image = denoise_median(noisy_image)
            progress_bar.progress(40)
            end = time.time()
            st.header('Imagine după aplicarea filtrului Median')

            st.image(denoised_image)
            st.success('PSNR-ul imaginii: %.3f db  ' % (PSNR(ground_truth, denoised_image)))

            progress_bar.progress(100)
            progress_bar.empty()

        elif model == 'Geometric':
            progress_bar = st.progress(0)
            start = time.time()
            progress_bar.progress(10)
            denoised_image = denoise_geometric(noisy_image)
            progress_bar.progress(40)
            end = time.time()
            st.header('Imagine după aplicarea filtrului Geometric')

            st.image(denoised_image)
            st.success('PSNR-ul imaginii: %.3f db  ' % (PSNR(ground_truth, denoised_image)))

            progress_bar.progress(100)
            progress_bar.empty()

@st.cache_data(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/SanzianuGabriela/sva/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def main():
    # print(cv2.__version__)

    selected_box = st.sidebar.selectbox(
        'Alege o opțiune..',
        ('Despre proiect', 'Algoritmi', 'Vezi codul sursa')
    )

    readme_text = st.markdown(get_file_content_as_string("README.md"))
    if selected_box == 'Despre proiect':
        st.sidebar.success('Pentru a vedea funcționalitatea algoritmilor apăsați butonul Algoritmi.')
    if selected_box == 'Algoritmi':
        readme_text.empty()
        models()
    if selected_box == 'Vezi codul sursa':
        readme_text.empty()
        st.code(get_file_content_as_string("main.py"))



if __name__ == "__main__":
    main()
