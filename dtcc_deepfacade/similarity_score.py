import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import functional as F
from PIL import Image
import os, argparse, pathlib, sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import numpy as np

project_folder_path = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_folder_path))

from dtcc_deepfacade.utils import find_files_in_folder

def generate_similarity_score_clusters(croped_images_dir):
    # Initialize the pre-trained model (e.g., ResNet-18)
    model = models.resnet152( weights=models.ResNet152_Weights.DEFAULT)
    model = model.eval()  # Set the model to evaluation mode

    # Define a transformation for image preprocessing
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Function to extract embeddings from an image
    def get_image_embedding(image_path, model, transform):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = model(image)
        return F.normalize(embedding, p=2, dim=1)  # L2-normalize the embeddings

    # Calculate KLDivergence between two probability distributions
    def calculate_kl_divergence(p, q):
        return torch.sum(p * (torch.log(p) - torch.log(q)))

    
        
    # List of image file paths
    image_paths = find_files_in_folder(croped_images_dir, extension='.jpg')

    # Compute embeddings for all images
    embeddings = [get_image_embedding(image_path, model, transform) for image_path in image_paths]

    # Calculate Kullback-Leibler Divergence between all pairs of images
    kl_divergence_matrix = torch.zeros((len(embeddings), len(embeddings)))

    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i != j:
                p = F.softmax(embeddings[i], dim=1)
                q = F.softmax(embeddings[j], dim=1)
                kl_divergence = calculate_kl_divergence(p, q)
                kl_divergence_matrix[i, j] = kl_divergence.item()

    labels = [os.path.split(image_path)[1].split('.')[0] for image_path in image_paths]



    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(kl_divergence_matrix, method='ward')

    # Calculate the cophenetic distances
    coph_dists = sch.cophenet(linkage_matrix)

    # Calculate the distances between cluster points
    distances = [linkage_matrix[i, 2] for i in range(linkage_matrix.shape[0])]

    # Use the "elbow method" to find the threshold
    diffs = np.diff(distances)
    elbow_point = np.argmax(diffs) + 1  # Adding 1 because the difference array is one element shorter

    # Set the threshold based on the elbow point
    threshold = distances[int(len(distances)/2 + len(distances)/4)]

    # # print("Threshold: ", threshold)
    # Set a threshold for cutting the dendrogram (you can adjust this)
    # threshold = 0.0014

    # Cut the dendrogram to form clusters
    cluster_ids = sch.fcluster(linkage_matrix,threshold,criterion = 'distance')

    # Create a dictionary to map clusters to image paths
    clustered_images = {}
    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id not in clustered_images:
            clustered_images[cluster_id] = []
        clustered_images[cluster_id].append(image_paths[i])

    # # print the clusters
    # for cluster_id, images in clustered_images.items():
    #     # print(f"Cluster {cluster_id}: {images}")

    save_path = os.path.join(croped_images_dir, "similarity_clusters")
    if not os.path.exists(save_path): os.mkdir(save_path)
    # Visualize the dendrogram using Seaborn
    sns.set()
    plt.figure(figsize=(20, 10))
    dendrogram = sch.dendrogram(linkage_matrix, labels=image_paths, orientation="right", leaf_font_size=8)
    plt.title("Dendrogram")
    plt.savefig(os.path.join(save_path, "kl_divergence_dendrogram_cluster.png"), dpi=300, bbox_inches='tight')


    def display_cluster_images(cluster_id, image_paths, clusters):
        cluster_images = [image_paths[i] for i, cluster in enumerate(clusters) if cluster == cluster_id]

        plt.figure(figsize=(20, 8))
        plt.suptitle(f"Cluster {cluster_id} - Images")
        for i, image_path in enumerate(cluster_images):
            plt.subplot(4, 4, i + 1)  # Adjust the number of rows and columns as needed
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')

        plt.savefig( os.path.join(save_path, f"cluster_{cluster_id}.png"), dpi=300, bbox_inches='tight')

    # Visualize the clusters
    unique_clusters = set(cluster_ids)

    for cluster_id in unique_clusters:
        display_cluster_images(cluster_id, image_paths, cluster_ids)


if __name__=='__main__':
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsdir', type=str, required=True, help='images folder path')
    args = parser.parse_args()
    generate_similarity_score_clusters(croped_images_dir=args.imgsdir)