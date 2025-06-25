#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:55:33 2025

@author: alex
"""

#%% Imports

import init
import h5py as h5
#from sys_neuro_tools import sleap_utils
import numpy as np 
import numpy.linalg
import matplotlib.pyplot as plt

def process_hdf5_data(filename):
    '''Returns a 'processed dictionary with information from the hdf5 file
    ---
    Params: filename (path for the hdf5 analysis file)
    ---
    Returns: dict with  the following keys
        locations  shape: shape of the locations data
        node_names: name of the nodes in the order they are found in locations
        dset_names: all the keys in original file
        locations: position data (#frames x #nodes x 2{x,y})
    '''
    with h5.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
        edge_inds = [edgeInd for edgeInd in f["edge_inds"]]
        #print(f["edge_inds"])

        
        frame_count, node_count, _, instance_count = locations.shape

        
        return {'locations_shape': locations.shape, 
                'node_names': node_names, 
                'dset_names': dset_names, 
                'locations': locations,
                'edge_inds': edge_inds}

#filename = r"/Users/alex/Downloads/ResultInference.hdf5"
# Open the HDF5 file in read mode ('r')
#new_dict = process_hdf5_data(filename)
"""for Key in new_dict:
    print(Key)
    print(np.shape(new_dict[Key]))    
print(new_dict["node_names"])"""
def PlotPorts(port_pos_list):
    for c_idx in range(port_pos_list.shape[1]):
        port_pos = port_pos_list[:,c_idx]
        plt.scatter(port_pos[0], port_pos[1], label=f"port {c_idx + 1}")
def PlotSkeleton(frame, processed_dict):
    '''Uses processed dict from process_hdf5_data to visualize the skeleton
    in matplotlib figure. DOES NOT SHOW GRAPH AUTOMATICALLY (use plt.show()).
    ---
    Params: 
    frame: the current frame in the video that needs to be plotted
    processed_dict: result of processing hdf5 file
    port1: position of top port
    port2: position of middle port
    port3: position of bottom port
    ---
    Returns: None
    '''
    plt.gca().invert_yaxis() #Images have 0,0 at top left and positive down
    for node_idx in range(len(frame)):
        plt.scatter(frame[node_idx,0], frame[node_idx,1], label=processed_dict["node_names"][node_idx])
        plt.legend()

    for edge_ind in processed_dict["edge_inds"]:
        #Get the first and second indices and use them to get the x position
        x = [frame[edge_ind[0]][0], frame[edge_ind[1]][0]]
        #Get the first and second indices and use them to get the y position
        y = [frame[edge_ind[0]][1], frame[edge_ind[1]][1]]
        #Plot the current edge
        plt.plot(x, y, color = "black")

def TranslationMatrix(point, inverted=False):
    #print(point)
    ans = None
    if(inverted):
        ans = [[1,0,float(-point[0])],
               [0,1,float(-point[1])]]
    else:
        ans = [[1,0,float(point[0])],
               [0,1,float(point[1])]]
    #print(ans)
    #print(np.shape(ans))
    return np.array(ans)

def BasisChangeMatrix(basis1, basis2):
    #Basis vectors must be normalized to prevent stretching
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    #basis vectors must be 2x1
    if np.shape(basis1) != (2,1):
        basis1=np.transpose(basis1)
    if np.shape(basis2) != (2,1):
        basis2=np.transpose(basis2)
    #Create change of basis matrix
    M = np.concatenate((basis1, basis2), axis=1) 
    M_old = [[1,0],[0,1]]
    change_of_basis_matrix = np.linalg.inv(M) @ M_old 
    return change_of_basis_matrix

def Angle(v1, v2):
    return np.arccos(np.dot(np.transpose(v1),v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def RotationMatrix(original_x_vect, new_x_vect):
    #Check to see if over or un
    print("Hi")
    cross_product = original_x_vect[0]*new_x_vect[1]-original_x_vect[1]*new_x_vect[0]
    #+ is rotated counterclockwise (Corrected clockwise to local)
    #- is rotated clockwise (Corrected counterclockise to local)
    angle = Angle(original_x_vect, new_x_vect)
    if cross_product > 0:
        return [[np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)]]
    else:
        return [[np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]]
    
    

def NodePositionsLocal(frame, processed_dict, origin_node="body", basis_node="neck", right_ortho=True):
    '''returns the node positions in a local coordinate system.
    The first bases vector is from the body to the neck.
    The second basis vector is orthogonal and on the right side of the body.
    ---
    Params: 
    processed_dict (result of processing hdf5 file)
    origin: the point (0,0)
    basis_node: the node used to define the first basis vector
    right_ortho: is the second basis vector on the right side of the body'''
    local_locations = np.zeros(np.shape(frame))
    #print(local_locations.shape)
    #get the positions of the body, neck
    origin_idx = processed_dict["node_names"].index(origin_node)
    basis_idx = processed_dict["node_names"].index(basis_node)
    p_origin = frame[origin_idx]
    p_basis = frame[basis_idx]
    #print("Origin")
    #print(p_origin)
    #print(np.shape(p_origin))        
    #print("Basis")
    #print(p_basis)
    #print(np.shape(p_basis))
    #Create basis vectors
    b1 = np.subtract(p_basis, p_origin)
    b2 = np.array([b1[1], -b1[0]]) 
    for n in range(len(frame)): 
        v = np.append(frame[n], 1)
        v = np.reshape(v, (3, 1))
        #print(v)
        #if n == origin_idx:
            #print("Original")
            #print(v)
            #print("Translational Matrix")
        centered_pos = np.dot(TranslationMatrix(p_origin, inverted=True), v)
        print((centered_pos + p_origin)[1] == frame[n][1])
        #if n == origin_idx:
            #print("Centered Pos")
            #print(centered_pos)
        new_pos = BasisChangeMatrix(b1, b2) @ centered_pos
        #if n == origin_idx: 
            #print("Basis Change Matrix")
            #print(BasisChangeMatrix(b1, b2))
            #print("New Pos")
            #print(new_pos)
        #frame[n][:] = np.reshape(new_pos, (2,1))
        local_locations[n][:] = np.reshape(new_pos, (2,1))
    return local_locations

def AngleToPorts(frame, processed_dict, port_pos_list):
    '''Returns the angle between the head-nose vector and all three ports
    ---
    Params:
    frame: the frame where the angles are being found
    processed_dict (result of processing hdf5 file)
    port_pos_list: a list of all the ports' positions (2xn numpy array)'''
    #KNOWN ERROR: When the nose disappears, the angle cannot be calculated
    #Use a.b = |a||b|cos(theta)
    angles = np.zeros((port_pos_list.shape[1], ))
    head_idx = processed_dict["node_names"].index("head")    
    nose_idx = processed_dict["node_names"].index("nose")
    p_head = frame[head_idx]
    p_nose = frame[nose_idx]
    #print("Head")
    #print(p_head)
    #print("Nose")
    #print(p_nose)
    nose_head_v = p_nose-p_head
    #print("Nose head vector:")
    #print(nose_head_v) 
    for c_idx in range(port_pos_list.shape[1]):
        port_pos = port_pos_list[:,c_idx]
        #Makes positions into column vectors for the next step
        port_pos = np.reshape(port_pos, (2,1)) 
        #print(f"Port {c_idx + 1}")
        #print(port_pos)
        #print(np.shape(port_pos))
        #Calculate vectors
        port_head_v = np.subtract(port_pos, p_head)
        #print("Port head vector")
        #print(port_head_v)
        #Use helper function to calculate angle b.w. two vectors
        angle = Angle(nose_head_v, port_head_v)
        angles[c_idx] = angle
    return angles

def PlotAnglesToPorts(angles, port_pos_list, offset, degrees=True):
    if len(angles) != port_pos_list.shape[1]:
        print("Error : Number of ports and angles do not match")
        return
    if degrees:
        for i in range (len(angles)):
            angles[i] = np.rad2deg(angles[i])
    for c_idx in range(port_pos_list.shape[1]):
        port_pos = port_pos_list[:,c_idx]
        if degrees:
            plt.text(port_pos[0]+offset[0], port_pos[1]+offset[1], f"{round(angles[c_idx],2)} degrees")
        else:
            plt.text(port_pos[0]+offset[0], port_pos[1]+offset[1], f"{round(angles[c_idx],2)} rad")

def PlotLocalPosNode(processed_dict, node_name, local_pos_list):
    print(np.shape(local_pos_list[:,processed_dict["node_names"].index(node_name)]))
    maxInd = len(local_pos_list)-1
    randomPoints = np.random.randint(low = 0, high = maxInd, size = (30,))
    for rp in randomPoints:
        coordinates = local_pos_list[rp, processed_dict["node_names"].index(node_name), :]
        plt.scatter(coordinates[0], coordinates[1])
    plt.xlabel("Body Axis Position (pixels)")
    plt.ylabel("Right Axis Position (pixels)")
    plt.suptitle(node_name)
    plt.show();
    
def ReformatToOriginal(df):
    num_nodes = (len(df.iloc[0])-5)//2 #4 components + cluster
    locations = np.zeros((len(df),num_nodes, 2))
    node_names = []
    for f_idx in range(len(df)):
        for n_idx in range(num_nodes):
            locations[f_idx, n_idx, 0] = df.iloc[f_idx][df.columns.values[n_idx*2]]
            locations[f_idx, n_idx, 1] = df.iloc[f_idx][df.columns.values[n_idx*2 + 1]]
            node_names.append(df.columns.values[n_idx*2][:-2])
    return {"node_names" : node_names,
            "locations" : locations}

def VelocityOutlierDetection(locations, num_std=3):
    '''Calculates differences in position over time and marks any suspiciously 
    rapid movement.
    
    Parameters
    ---
    locations: locations data from hdf5 file(frames x nodes x (x,y) x 1)
    
    Returns
    ---
    flags: same shape with 0 for normal and 1 for outliers'''
    mask = np.zeros(np.shape(locations))
    for n_idx in range(locations.shape[1]):
        x_list = locations[:,n_idx,0,0]
        y_list = locations[:,n_idx,1,0]
        
        x_dIff = np.diff(x_list)
        y_diff = np.diff(y_list)
        
        x_std = np.std(x_dIff)
        y_std = np.std((y_diff))
        
        x_mean = np.mean(x_dIff)
        y_mean = np.mean(y_diff)
        
        #Creates bounds num_std standard deviations above/below the mean
        x_bounds = [x_mean-x_std*num_std, x_mean+x_std*num_std]
        y_bounds = [y_mean-y_std*num_std, y_mean+y_std*num_std]
        
        #Checks those bounds at all points
        for frame_idx in range(locations.shape[0]):
            mask[frame_idx, n_idx, 0,0] = (locations[frame_idx, n_idx, 0,0] < x_bounds[0] or locations[frame_idx, n_idx, 0,0] > x_bounds[1])
            mask[frame_idx, n_idx, 1,0] = (locations[frame_idx, n_idx, 1,0] < y_bounds[0] or locations[frame_idx, n_idx, 1,0] > y_bounds[1])
    return mask
        
        
        
            
            

        
        


        
        
