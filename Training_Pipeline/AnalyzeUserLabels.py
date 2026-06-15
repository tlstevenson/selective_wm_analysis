# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scikit-learn",
#     "scipy",
#     "sleap-io",
# ]
# ///

import sleap_io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def center_and_clean_pose(points):
    """Centers a pose to the origin and handles missing nodes."""
    # Find valid (labeled) points
    valid_mask = ~np.isnan(points).any(axis=1)
    valid_points = points[valid_mask]
    
    if len(valid_points) == 0:
        return None
        
    # Center the pose by subtracting the centroid of the valid points
    centroid = np.mean(valid_points, axis=0)
    centered_points = points - centroid
    
    # Fill missing points (NaNs) with 0,0 (the new centroid) 
    # so they don't break the clustering distance metrics
    centered_points = np.nan_to_num(centered_points, nan=0.0)
    return centered_points

def main(slp_file_path, max_k=10):
    # 1. Load the SLEAP project using sleap-io
    print(f"Loading {slp_file_path}...")
    labels = sio.load_slp(slp_file_path)
    
    if not labels.skeletons:
        raise ValueError("No skeletons found in the provided .slp file.")
        
    skeleton = labels.skeletons[0]
    
    # Extract node names for labeling the plot
    node_names = [node.name for node in skeleton.nodes]
    
    # Extract edge indices mapping for plotting lines
    edges = []
    for edge in skeleton.edges:
        src_idx = skeleton.nodes.index(edge.source)
        dst_idx = skeleton.nodes.index(edge.destination)
        edges.append((src_idx, dst_idx))
    
    # 2. Extract user instances
    poses = []
    pose_arrays = []
    
    for lf in labels:
        for instance in lf.instances:
            # Strictly filter for manual User Instances (ignores PredictedInstance)
            if type(instance) is sio.Instance:
                # Get the (n_nodes, 2) coordinate array
                pts = instance.numpy() 
                processed_pts = center_and_clean_pose(pts)
                
                if processed_pts is not None:
                    poses.append(processed_pts.flatten()) # Flatten for K-Means
                    pose_arrays.append(processed_pts)     # Keep original shape for plotting

    X = np.array(poses)
    if len(X) < 3:
        raise ValueError("Not enough user instances to cluster.")

    print(f"Extracted {len(X)} valid user-labeled poses.")

    # 3. Determine the optimal number of clusters using Silhouette Score
    best_k = 2
    best_score = -1
    max_k = min(max_k, len(X) - 1)
    
    print("Evaluating cluster counts...")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Optimal number of unique clusters found: {best_k} (Silhouette Score: {best_score:.3f})")

    # 4. Final clustering with the optimal K
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto').fit(X)
    cluster_labels = kmeans.labels_
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    # 5. Check for overrepresented clusters
    expected_count = len(X) / best_k
    overrep_threshold = expected_count * 1.5 # Flag if a cluster is 50% larger than uniform expectation
    
    print("\n--- Cluster Distribution ---")
    for i, count in zip(unique, counts):
        percentage = (count / len(X)) * 100
        status = "[OVERREPRESENTED]" if count > overrep_threshold else ""
        print(f"Cluster {i}: {count} poses ({percentage:.1f}%) {status}")

    # 6. Plot the median example for each cluster
    fig, axes = plt.subplots(1, best_k, figsize=(5 * best_k, 6))
    if best_k == 1:
        axes = [axes]
        
    for i in range(best_k):
        # Find the pose closest to the cluster center
        centroid = kmeans.cluster_centers_[i]
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_poses = X[cluster_indices]
        
        # Calculate Euclidean distance from all cluster poses to the centroid
        distances = cdist([centroid], cluster_poses)
        median_local_idx = np.argmin(distances)
        median_global_idx = cluster_indices[median_local_idx]
        
        # Get the (n_nodes, 2) shaped array for plotting
        median_pose = pose_arrays[median_global_idx]
        
        ax = axes[i]
        
        # Plot edges
        for edge in edges:
            src, dst = edge
            ax.plot([median_pose[src, 0], median_pose[dst, 0]], 
                    [median_pose[src, 1], median_pose[dst, 1]], 
                    'k-', alpha=0.5)
            
        # Plot nodes
        ax.scatter(median_pose[:, 0], median_pose[:, 1], c='red', zorder=5)
        
        # Add text labels for every node
        for j, node_name in enumerate(node_names):
            ax.annotate(node_name, 
                        (median_pose[j, 0], median_pose[j, 1]),
                        textcoords="offset points",
                        xytext=(3, 3), # Offset slightly right and down (since y is inverted)
                        ha='left',
                        fontsize=7,
                        color='darkblue',
                        zorder=6)
        
        # Formatting
        ax.set_title(f"Cluster {i}\n(N={counts[i]})")
        ax.invert_yaxis() # Match image coordinates where Y goes down
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle("Median Representative Pose per Cluster", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    SLP_FILE = r"C:\Users\cns-th-lab\SLEAP_Projects\postsurgery_all_vid_bulky.slp"
    main(SLP_FILE)