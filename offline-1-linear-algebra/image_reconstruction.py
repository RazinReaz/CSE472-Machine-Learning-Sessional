import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approximation(A, k):
    U, S, V_T = np.linalg.svd(A)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_T_k = V_T[:k, :]
    A_k = U_k.dot(S_k).dot(V_T_k)
    return A_k

if __name__ == "__main__":
    image = cv.imread("image.JPG", cv.IMREAD_GRAYSCALE)
    # show the imgae 
    # cv.imshow("image", image)
    # cv.waitKey(0)
    image = cv.resize(image, (1000, 1300))
    image = np.array(image)
    k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200, 500, 900]
    c = 3
    fig, ax = plt.subplots(len(k_list)//c, c, figsize=(15, 15))

    for  i, k in enumerate(k_list):
        reconstructed_image = low_rank_approximation(image, k)
        ax[i//c, i%c].imshow(reconstructed_image, cmap='gray')
        ax[i//c, i%c].set_title(f"k = {k}")
    # save plot to pdf
    plt.savefig("image_reconstruction.jpg")