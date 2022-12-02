# The following functions are the same you implemented the past week
# You do not need to edit the functions of this code block (just run it)
def convert_to_shape_pixels_by_bands(data):
    num_dimensions = len(data.shape)
    assert(num_dimensions == 2 or num_dimensions == 3)
    if num_dimensions == 3:
        num_bands = data.shape[2]
        return data.reshape((-1, num_bands))
    else:
        return data

def compute_average_feature(data):
    # If needed convert data to the shape (num_pixels x num_bands)
    data_2d = convert_to_shape_pixels_by_bands(data)
    # Get the number of bands
    num_bands = data_2d.shape[1]
    avg_features = np.zeros(num_bands)
    for b in range(num_bands):
        # Compute the average value of each band (use the function np.mean)
        avg_features[b] = np.mean(data_2d[:, b])
    return avg_features

def compute_standard_deviation_feature(data):
    # If needed convert data to the shape (num_pixels x num_bands)
    data_2d = convert_to_shape_pixels_by_bands(data)
    # Compute the standard deviation feature (using the numpy function np.std)
    #       as in the function compute_average_feature iterate over the bands
    #       and compute one value for each band
    num_bands = data_2d.shape[1]
    avg_features = np.zeros(num_bands)
    for b in range(num_bands):
        avg_features[b] = np.std(data_2d[:, b])
    return avg_features

def compute_histogram_feature(data, num_bins=10):
    # If needed convert data to the shape (num_pixels x num_bands)
    data_2d = convert_to_shape_pixels_by_bands(data)
    num_bands = data_2d.shape[1]
    hist_features = np.zeros((num_bands, num_bins)).astype(np.float32)
    for b in range(num_bands):
        # Compute the histogram for each band 
        #       use the function np.histogram(array, bins=num_bins)
        hist, boundaries = np.histogram(data_2d[:, b], bins=num_bins)
        hist_features[b, :] = hist
    # Return a 1D array containing all the values
    return hist_features.flatten()

def compute_image_features_from_regions(image, segmentation_map):
    num_regions = len(np.unique(segmentation_map))
    all_features = []
    for id_region in range(num_regions):
        # Obtain pixel values of each regions, with shape (num_pixels x num_bands)
        pixel_values = image[segmentation_map==id_region]
        # Compute the average, standard deviation and histogram features
        #       and concatenated them unsing the function (np.concatenate)
        avg = compute_average_feature(pixel_values)
        features = compute_standard_deviation_feature(pixel_values)
        hist_features = compute_histogram_feature(pixel_values)
        features = np.concatenate([avg, features, hist_features])
        # Add concatenated features to the variable all_features
        all_features.append(features)
    # convert list to numpy array of shape: (num_regions x num_bands)
    return np.array(all_features).astype(np.float32)



def get_label_per_region(segmented_image, label_map):
    """
    Returns a 1D numpy array that contains the label for each region, shape: (num_regions)
            For each region, we obtain the label that has the largest intersection with it
    """
    num_regions = len(np.unique(segmented_image))
    num_labels = len(np.unique(label_map))
    region_labels = []
    for region_id in range(num_regions):
        mask_region = segmented_image == region_id
        
        intersection_per_label = []
        for label_id in range(num_labels):
            mask_label = label_map == label_id
            # Compute intersection of each region with each label
            intersection = np.sum(mask_region * mask_label)
            intersection_per_label.append(intersection)
        
        intersection_per_label = np.array(intersection_per_label)
        # Obtain the index of the label with largest intersection
        selected_label = np.argmax(intersection_per_label)
        region_labels.append(selected_label)
    
    return np.array(region_labels).astype(np.uint32)
import numpy as np
from skimage.io import imsave, imread
from sklearn.ensemble import RandomForestClassifier


def RF_feature_region(output,target):
    # Create arrays of training targets and features 
    all_train_region_features = []
    all_train_region_labels = []

    for image in output:
        # TODO: read segmented image
        segmented_image = slic(image, n_segments=1000, start_label=0)
        # TODO: read ground truth image
        gt_image = target
        # TODO: Get labels per region using the function "get_label_per_region" defined above
        region_labels = get_label_per_region(segmented_image, gt_image)
        # Add current region labels to the variable all_train_region_labels
        all_train_region_labels.append(region_labels)
        # TODO: compute features 
        region_features = compute_image_features_from_regions(image, segmented_image)
        # Add current region features to the variable all_train_region_features
        all_train_region_features.append(region_features)

    # Tranforming the list all_train_region_labels in an array of shape: (num_all_regions)
    train_labels = np.concatenate(all_train_region_labels)
    print("train_labels shape " + str(train_labels.shape))
    # Tranforming the list all_train_region_features in an array of shape: (num_all_regions, num_features)
    train_features = np.concatenate(all_train_region_features)
    print("train_features shape " + str(train_features.shape))
    mean_per_feature = np.mean(train_features, axis=0)
    std_per_feature = np.std(train_features, axis=0)
    # TODO: normalize features by substracting the mean and dividing by the standard deviation
    norm_train_features = (train_features - mean_per_feature) / std_per_feature

    # TODO: create random forest classifier with the parameter random_state=10
    classifier = RandomForestClassifier(random_state=10)
    # TODO: fit the model with the normalized features
    classifier.fit(norm_train_features, train_labels)
    

    
import torch
def RF_CNN(output,target,nbr_class):
    # Create arrays of training targets and features 
    all_train_features = []
    all_train_labels = []

    for i in range(len(output)):
        #print(output.shape) #[batch,features,200,200]
        #print(target.shape)#[batch,1,200,200] 
        label_image = target[i].flatten() #[200*200]
        all_train_labels.append(label_image.cpu())
        # TODO: compute features 
        train_features_pixel = output[i].flatten(1).permute(1,0) #[200*200,features]
        # Add current region features to the variable all_train_region_features
        all_train_features.append(train_features_pixel.cpu())

    # Tranforming the list all_train_region_labels in an array of shape: (num_all_regions)
    train_labels = torch.cat(all_train_labels)

    # Tranforming the list all_train_region_features in an array of shape: (num_all_regions, num_features)
    train_features = torch.cat(all_train_features)

    mean_per_feature = torch.mean(train_features, axis=0)
    std_per_feature = torch.std(train_features, axis=0)
    # TODO: normalize features by substracting the mean and dividing by the standard deviation
    norm_train_features = (train_features - mean_per_feature) / std_per_feature

    # TODO: create random forest classifier with the parameter random_state=10
    classifier = RandomForestClassifier(random_state=10, n_estimators=20, max_depth=5)
    # TODO: fit the model with the normalized features
    classifier.fit(norm_train_features.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
                   
    # Predict label of regions 
    label_predictions = classifier.predict(norm_train_features.cpu().detach().numpy()) #[batch*200*200]
    # Compute label predictions per pixel
    #desire predicion_map shape  #[batch*200*200,class]
    predicion_map = torch.zeros((output.shape[0]*output.shape[2]*output.shape[3],nbr_class))
    for j in range(len(predicion_map)):
        predicion_map[j,label_predictions[j]] = label_predictions[j]
    
    print("RF_done")
    print(torch.tensor(predicion_map))
    predicion_map = torch.tensor(predicion_map,requires_grad=True)
    predicion_map = predicion_map.to("cuda")
    return predicion_map






























