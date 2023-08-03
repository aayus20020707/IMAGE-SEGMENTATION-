import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from skimage import io, img_as_float
from scipy.ndimage import label
import os

def load_image(image_path):
    # Load the image using skimage and convert it to floating-point values
    image = img_as_float(io.imread(image_path))
    return image

def kmeans_segmentation(image, n_clusters=2, random_state=42):
    # Flatten the image to a 2D array of pixels
    rows, cols, _ = image.shape
    pixels = image.reshape(rows * cols, -1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(pixels)

    # Replace pixel values with cluster centers
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(rows, cols, -1)

    return segmented_image

def agglomerative_segmentation(image, n_clusters=2):
    # Flatten the image to a 2D array of pixels
    rows, cols, _ = image.shape
    pixels = image.reshape(rows * cols, -1)

    # Apply Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative.fit_predict(pixels)

    # Replace pixel values with cluster centers (mean of each cluster)
    segmented_image = np.array([np.mean(pixels[labels == label], axis=0) for label in range(n_clusters)])
    segmented_image = segmented_image[labels].reshape(rows, cols, -1)

    return segmented_image

def visualize_segmentation(original_image, segmented_image, n_clusters):
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f"Segmented Image (K={n_clusters})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def save_segmented_image(segmented_image, output_path):
    io.imsave(output_path, segmented_image)

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with the path to your image file
    n_clusters_kmeans = 2  # Number of clusters for K-means
    n_clusters_agglo = 3  # Number of clusters for Agglomerative Clustering

    # Load the image
    image = load_image(image_path)

    # Perform image segmentation using K-means
    segmented_image_kmeans = kmeans_segmentation(image, n_clusters=n_clusters_kmeans)

    # Perform image segmentation using Agglomerative Clustering
    segmented_image_agglo = agglomerative_segmentation(image, n_clusters=n_clusters_agglo)

    # Visualize the original and segmented images
    visualize_segmentation(image, segmented_image_kmeans, n_clusters_kmeans)
    visualize_segmentation(image, segmented_image_agglo, n_clusters_agglo)

    # Save the segmented images to files
    output_dir = "segmented_images"
    os.makedirs(output_dir, exist_ok=True)

    save_segmented_image(segmented_image_kmeans, os.path.join(output_dir, "segmented_kmeans.jpg"))
    save_segmented_image(segmented_image_agglo, os.path.join(output_dir, "segmented_agglo.jpg"))
