# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 09:50:29 2025

@author: hankslab(simran)
"""

#%% Imports

import init

import pyutils.utils as utils
from sys_neuro_tools import ephys_utils
from sys_neuro_tools import sleap_utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Get HDF5 data

filename = r"C:\Users\hankslab\SLEAP\SLEAP_Test 2\rat_v3\test.000_Test_0002_reformat.analysis.h5"
node_locations_data = sleap_utils.get_hdf5_data(filename)

filename = r"C:\Users\hankslab\SLEAP\SLEAP_Test 2\ports\ports.000_Test_0002_reformat.analysis.h5"
port_locations_data = sleap_utils.get_hdf5_data(filename)


#%% Smooth animal nodes and average port locations

node_locations = node_locations_data['locations'][:,:,:,0]

#subset of body potitions data
start_index = 0
end_index = 100
subset_node_locations = node_locations[start_index:end_index,:,:]

smoothed_node_locations = sleap_utils.get_smoothed_body_positions(subset_node_locations, 9)

port_locations = port_locations_data['locations'][:,:,:,0]
mean_values = np.nanmean(port_locations, axis=0)
port_locations[:] = mean_values

subset_port_locations = port_locations[start_index:end_index,:,:]


#%% Plot animal nodes and smoothed animal nodes

num_frames = subset_node_locations.shape[0]
num_nodes = subset_node_locations.shape[1]

fig, axs = plt.subplots (nrows=4, ncols=3, figsize=(15,10))

axs = axs.flatten()

for i in range(num_nodes):
    axs[i].plot(range(num_frames), subset_node_locations[:,i,0], label='Original X')
    axs[i].plot(range(num_frames), subset_node_locations[:,i,1], label='Original Y')
    
    axs[i].plot(range(num_frames), smoothed_node_locations[:,i,0], label='Smoothed X')
    axs[i].plot(range(num_frames), smoothed_node_locations[:,i,1], label='Smoothed Y')
    
    axs[i].set_title(f'Node {i+1}')
    axs[i].set_xlabel('Frame Number')
    axs[i].set_ylabel('Position')
    axs[i].legend()
    axs[i].grid()
    
plt.tight_layout()
plt.show()


#%% Calculate vectors and create dataframe

head_nose_vectors = np.zeros((num_frames, 2))

for frame in range(num_frames):
    head_position = smoothed_node_locations[frame, 6, :]
    nose_position = smoothed_node_locations[frame, 0, :]
    
    head_nose_vectors[frame] = nose_position - head_position

        
r = np.sqrt(head_nose_vectors[:,0]**2 + head_nose_vectors[:,1]**2)
theta = np.arctan2(head_nose_vectors[:,1], head_nose_vectors[:,0])

head_nose_polar = np.zeros((head_nose_vectors.shape[0], 2))
head_nose_polar[:,0] = r
head_nose_polar[:,1] = theta

head_ports_vectors = np.zeros((num_frames, 3, 2))
num_ports = subset_port_locations.shape[1]

for frame in range(num_frames):  
    head_position = smoothed_node_locations[frame, 6, :]
    
    for port in range(num_ports):
        port_position = subset_port_locations[frame, port, :]
        
        head_ports_vectors[frame, port] = port_position - head_position
        
head_ports_polar = np.zeros((num_frames, num_ports, 2))

for frame in range(num_frames):
    for port in range(num_ports):
        x, y = head_ports_vectors[frame, port, :]
        d = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        head_ports_polar[frame, port, 0] = d
        head_ports_polar[frame, port, 1] = theta
        
#angles between the head_nose and head_port vectors
head_nose_thetas = head_nose_polar[:,1]
head_ports_thetas = head_ports_polar[:,:,1]

theta_right = head_ports_thetas[:,0] - head_nose_thetas
theta_center = head_ports_thetas[:,1] - head_nose_thetas
theta_left = head_ports_thetas[:,2] - head_nose_thetas


nodes_ports_data = {}

for i, node in enumerate(node_locations_data['node names'][:11]):
    nodes_ports_data[f'{node} x'] = smoothed_node_locations[:, i, 0]
    nodes_ports_data[f'{node} y'] = smoothed_node_locations[:, i, 1]
    
for i, port in enumerate(port_locations_data['node names']):
    nodes_ports_data[f'{port} x'] = subset_port_locations[:, i, 0]
    nodes_ports_data[f'{port} y'] = subset_port_locations[:, i, 1]
    
nodes_ports_df = pd.DataFrame(nodes_ports_data)

theta_d_data = { 
    'theta left': theta_left,
    'd left': head_ports_polar[:,0,0],
    'theta center': theta_center,
    'd center': head_ports_polar[:,1,0],
    'theta right': theta_right,
    'd right': head_ports_polar[:,2,0]
    }

theta_d_df = pd.DataFrame(theta_d_data)

nodes_ports_df = pd.concat([nodes_ports_df, theta_d_df], axis=1)

#%% Plot vectors

frame_index = 0

skeleton_nodes = {
    'Nose': smoothed_node_locations[frame_index, 0],
    'Head': smoothed_node_locations[frame_index, 6],
    'Left Ear': smoothed_node_locations[frame_index, 1],
    'Right Ear': smoothed_node_locations[frame_index, 2],
    'Neck': smoothed_node_locations[frame_index, 3],
    'Left Front': smoothed_node_locations[frame_index, 9],
    'Right Front': smoothed_node_locations[frame_index, 10],
    'Body': smoothed_node_locations[frame_index, 4],
    'Left Rear': smoothed_node_locations[frame_index, 7],
    'Right Rear': smoothed_node_locations[frame_index, 8],
    'Tail Start': smoothed_node_locations[frame_index, 5]
    }

plt.figure(figsize=(10,6))


plt.scatter(subset_port_locations[frame_index, :, 0], subset_port_locations[frame_index, :, 1], 
            color='red', label='Ports')

for name, position in skeleton_nodes.items():
    plt.scatter(position[0], position[1], label=name)
    plt.text(position[0] + 1, position[1] + 1, name, fontsize=9)
    
edges = [
    ('Nose', 'Head'),
    ('Head', 'Neck'),
    ('Neck', 'Left Ear'), ('Neck', 'Right Ear'),
    ('Body', 'Left Front'), ('Body', 'Right Front'),
    ('Neck', 'Body'),
    ('Body', 'Left Rear'), ('Body', 'Right Rear'),
    ('Body', 'Tail Start')
    ]

for start, end in edges:
    start_pos = skeleton_nodes[start]
    end_pos = skeleton_nodes[end]
    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='black', linewidth=1)

head_position = skeleton_nodes['Head']
plt.quiver(head_position[0], head_position[1], head_nose_vectors[frame_index, 0], head_nose_vectors[frame_index, 1], 
           angles='xy', scale_units='xy', scale=1, color='blue', label='Head-Nose Vector')

plt.quiver(np.full(3, head_position[0]), np.full(3, head_position[1]),
           head_ports_vectors[frame_index,:,0], head_ports_vectors[frame_index,:,1],
           angles='xy', scale_units='xy', scale=1, color='green', label='Head-Ports Vectors')

port_labels = ['Left Port', 'Center Port', 'Right Port']
for i in range(3):
    x = subset_port_locations[frame_index, i, 0]
    y = subset_port_locations[frame_index, i, 1]
    plt.text(x + 5, y + 5, port_labels[i], fontsize=10, color='red')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Rat Skeleton and Port Vectors')
plt.legend()
plt.grid(True)
plt.xlim(0, 1100)
plt.ylim(-250, 1000)

plt.show()



    
    