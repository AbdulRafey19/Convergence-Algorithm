
import sys
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch as th
import numpy as np
import open3d as o3d
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al


def main(ideal_file_path,defected_file_path):
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")
    fixed_image=al.read_image_as_tensor(ideal_file_path,dtype=dtype)
    moving_image=al.read_image_as_tensor(defected_file_path,dtype=dtype,device=device)
    fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    transformation = al.transformation.pairwise.SimilarityTransformation(moving_image, opt_cm=True)
    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.001, amsgrad=True)




    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(30)


    # start the registration
    registration.start()

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    end = time.time()

    print("=================================================================")

    print("Registration done in:", end - start, "s")
    print("Result parameters:")
    transformation.print()

    # plot the results
    plt.subplot(131)
    plt.xlabel('Image size scale')
    plt.ylabel('Image size scale')
    plt.imshow(fixed_image.numpy(), cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(132)
    plt.xlabel('Image size scale')
    plt.imshow(moving_image.numpy(), cmap='gray')
    plt.title('Moving Image')

    plt.subplot(133)
    plt.xlabel('Image size scale')
    plt.imshow(warped_image.numpy(), cmap='gray')
    plt.title('Warped Moving Image')

    plt.show()

    return fixed_image.numpy(), warped_image.numpy()

def ssim_cal(file_pathA,file_pathB,save_path,threshold=0.95,blur=False,target_size=[950,550]):
    
    
    import cv2
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    import matplotlib.pyplot as plt

    
    image1=file_pathA
    image2=file_pathB
    
    

    # Compute SSIM map
    ssim_val,ssim_map = ssim(image1, image2,full=True,data_range=1.0)
    print(f"SSIM Value is {ssim_val*100}")
    user_preference='yes'
    while (user_preference=='yes'):
        tolerance_factor=float(input('Input a pixel tolerance factor between 0.1 to 0.9'))
        # Set a threshold to identify pixels with high dissimilarity
        
        # Threshold higher is to identify more dissimilarity
        high_dissimilarity_mask = ssim_map > tolerance_factor
        

        high_dissimilarity_mask_visual = (high_dissimilarity_mask * 255).astype(np.uint8)
        
        plt.imshow(high_dissimilarity_mask_visual, cmap='jet', alpha=0.5)  # Overlay the dissimilarity regions
        plt.title('High Dissimilarity Regions')
        plt.axis('off')
        # Create a dummy plot element for the legend
        legend_patch = mpatches.Patch(color='cyan', label=f'Pixel SSIM values with tolerance factor {tolerance_factor}')

        # Add the legend at the bottom
        plt.legend(handles=[legend_patch], loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
        
        # Show the plots
        plt.tight_layout()
        plt.show()
        
        user_preference=input('do u want to test another tolerence factor, yes or no')

    # Example 2x2 array
    data = np.array(ssim_map)

    # Plot heatmap

    plt.imshow(data, cmap='hot', interpolation='nearest')
    colorbar = plt.colorbar()
    plt.title(f"SSIM Value is {ssim_val}")
    plt.xlabel('Image size scale')
    plt.ylabel('Image size scale')
    colorbar.set_label('SSIM')

    # Format SSIM value with 2 decimal places and multiply by 100
    ssim_val_formatted = "{:.2f}".format(ssim_val * 100)

    # Construct the filename with the formatted SSIM value
    new = f'with SSIM {ssim_val_formatted}.png'

    # Concatenate the save path and the filename
    full_save_path = save_path+ new

    plt.savefig(full_save_path)
    plt.show()
    return len(image2.flat), (ssim_val*100)


    
# Function to extract frame number from file path
def extract_frame_number(file_path):
    parts = file_path.split('_frame')
    
    if len(parts) == 2:
        frame_number = parts[1].split('.')[0]
        return frame_number
    else:
        return None





def delete_image(image_path):
    try:
        os.remove(image_path)
        print(f"Image at '{image_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Image file at '{image_path}' not found.")
    except Exception as e:
        print(f"An error occurred while deleting the image: {e}")


def max_pooling(image, pool_size=(2, 2)):
    import numpy as np
    """
    Perform max pooling on the given image.

    Parameters:
    - image: Input image as a numpy array.
    - pool_size: Size of the pooling window, default is (2, 2).

    Returns:
    - pooled_image: Image after max pooling.
    """

    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # Get image dimensions
    height, width = image.shape[:2]

    # Get pooling window size
    pool_height, pool_width = pool_size

    # Calculate new dimensions after pooling
    new_height = height // pool_height
    new_width = width // pool_width

    # Initialize pooled image array
    pooled_image = np.zeros((new_height, new_width), dtype=image.dtype)

    # Perform max pooling
    for i in range(new_height):
        for j in range(new_width):
            # Define pooling window boundaries
            start_row = i * pool_height
            end_row = start_row + pool_height
            start_col = j * pool_width
            end_col = start_col + pool_width

            # Extract the region of interest from the image
            roi = image[start_row:end_row, start_col:end_col]

            # Apply max pooling to the region of interest
            pooled_image[i, j] = np.max(roi)

    return pooled_image

def image_preprocesing(image1,image2,size=[5000,3000],blur=False,crop=0):
    '''
    Resizes the image
    setting blue=True applies median Blur
    
    '''

    image1=cv2.resize(image1,size)
    image2=cv2.resize(image2,size)
    

    # Calculate the crop boundaries
    crop_pixels = int(size[1] * crop)
    top_crop = crop_pixels
    bottom_crop = size[1] - crop_pixels

    # Crop the images
    image1 = image1[top_crop:bottom_crop, :]
    image2 = image2[top_crop:bottom_crop, :]

 
    if blur==True:
        


        medianBlurRealImg = cv2.medianBlur(image1,11)
        medianBlurSimImg = cv2.medianBlur(image2,11)
        image1=medianBlurRealImg
        image2=medianBlurSimImg




    return image1, image2

def accuracy_calc(predicted_pixels,total_no_of_pixels,actual_percentage_similarity):
    predicted_percentage_defect=(predicted_pixels/total_no_of_pixels)*100
    
    if (predicted_percentage_defect>(100-actual_percentage_similarity)):
        print(f"accuracy (over) :  {((predicted_percentage_defect-100+actual_percentage_similarity)*100)/(100-actual_percentage_similarity)}" )
    
    else:
        print(f"accuracy :  {(predicted_percentage_defect*100)/(100-actual_percentage_similarity)}" )

def compare_defects_and_ideal(image1, image2):
    

    from skimage.metrics import structural_similarity as ssim

    # Load the images (assuming they are already grayscale)
    image1 = image1
    image2 = image2

    
    image = image1
    mask = image2

    # Resize the mask image to match the size of the main image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    ret, binary_mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    bt = binary_mask

    # Create masks for white regions in the binary thresholded image
    white_mask = bt == 255

    # Extract pixels from image1 and image2 corresponding to white regions
    white_pixels_image1 = np.zeros_like(image1)
    white_pixels_image2 = np.zeros_like(image2)

    # Apply mask to image1 and image2
    white_pixels_image1[white_mask] = image1[white_mask]
    white_pixels_image2[white_mask] = image2[white_mask]


    

    print("COMPUTING METRIC--------------------------------")

    # Use grayscale images directly for SSIM calculation
    similarity_index, ssim2_map = ssim(white_pixels_image1, white_pixels_image2, full=True, data_range=255.0)
    
    user_preference='yes'
    while (user_preference=='yes'):
        tolerance_factor=float(input('Input a pixel tolerance factor between 0.1 to 0.9'))
        # Set a threshold to identify pixels with high dissimilarity
        #
        high_dissimilarity_mask = ssim2_map > tolerance_factor
        #print(len(high_dissimilarity_mask))
        # Optionally, visualize the high dissimilarity mask

        high_dissimilarity_mask_visual = (high_dissimilarity_mask * 255).astype(np.uint8)
        plt.imshow(high_dissimilarity_mask_visual, cmap='jet', alpha=0.5)  # Overlay the dissimilarity regions
        plt.title('High Dissimilarity Regions')
        plt.axis('off')
        # Create a dummy plot element for the legend
        legend_patch = mpatches.Patch(color='cyan', label=f'Pixel SSIM values with tolerance factor {tolerance_factor}')

        # Add the legend at the bottom
        plt.legend(handles=[legend_patch], loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.show()
        user_preference=input('do u want to test another tolerence factor, yes or no')

    data=np.array(ssim2_map)
    plt.title("SSIM heat map visualization")
    plt.xlabel("Image size scale")
    plt.ylabel("Image size scale")
    plt.imshow(data, cmap='hot', interpolation='nearest')
    colorbar = plt.colorbar()
    plt.title(f"SSIM Value is {similarity_index}")
    colorbar.set_label('SSIM')
    plt.show()

    print("SIMILARITY B/W predicted and ideal image " + str(float(similarity_index)))

    # Create visualizations (optional)
    target_size = (950, 550)  # Adjust target size if needed

    # Resize for visualization (optional)
    gray1 = cv2.resize(white_pixels_image1, target_size)
    gray2 = cv2.resize(white_pixels_image2, target_size)

    
    num_true = np.count_nonzero(white_mask)


    return num_true


def Grayscale_capture(ideal_path, rot_path, alldef_path,a,b,c,d,e,f):
    ideal_png = input("ENTER ADDRESS TO SAVE THE IDEAL POINT CLOUD AS PNG (do add .png after file name): ")
    alldef_png = input("ENTER ADDRESS TO SAVE THE TRUE DEFECT CLOUD AS PNG(do add .png after file name): ")
    rotdef_png = input("ENTER ADDRESS TO SAVE THE DEFECTED POINT CLOUD AS PNG (do add .png after file name): ")

    #FOR IDEAL
    df = pd.read_csv(ideal_path, delimiter=';', header=None)

    # Save the data in XYZ format
    df.to_csv('ideal.xyz', header=False, index=False, sep=' ')

    

    # Load the point cloud
    pcd = o3d.io.read_point_cloud("ideal.xyz")
    #os.remove('ideal.xyz')
    # Get the Z-coordinates (or any other scalar field)
    z_coords = np.asarray(pcd.points)[:, 2]  # Use Z for height-based grayscale

    # Normalize the Z-coordinates to the range [0, 1]
    z_min = z_coords.min()
    z_max = z_coords.max()
    grayscale = (z_coords - z_min) / (z_max - z_min)

    # Convert normalized Z to grayscale colors
    pcd.colors = o3d.utility.Vector3dVector(np.tile(grayscale[:, np.newaxis], (1, 3)))
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    #vis.create_window(visible=False)
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Set the background color to blue (RGB values for blue)
    vis.get_render_option().background_color = np.asarray([0, 0, 1])  # Blue background

    # Set the view control parameters for a top-down view
    view_control = vis.get_view_control()
    view_control.set_front([a, b, c])  # Set the front direction
    view_control.set_up([d, e, f])    # Set the up direction

    # Optionally, you can set the zoom level
    view_control.set_zoom(0.7)
    # Render the scene
    vis.run()
    


    # Capture the screen image
    vis.capture_screen_image(ideal_png)

    # Destroy the window to free up resources
    vis.destroy_window()

    #FOR ALL DEFECT
    df = pd.read_csv(alldef_path, delimiter=';', header=None)

    # Save the data in XYZ format
    df.to_csv('alldef.xyz', header=False, index=False, sep=' ')

    # Load the point cloud
    
    pcd = o3d.io.read_point_cloud("alldef.xyz")
    os.remove('alldef.xyz')
    # Get the Z-coordinates (or any other scalar field)
    z_coords = np.asarray(pcd.points)[:, 2]  # Use Z for height-based grayscale

    # Normalize the Z-coordinates to the range [0, 1]
    z_min = z_coords.min()
    z_max = z_coords.max()
    grayscale = (z_coords - z_min) / (z_max - z_min)

    # Convert normalized Z to grayscale colors
    pcd.colors = o3d.utility.Vector3dVector(np.tile(grayscale[:, np.newaxis], (1, 3)))
    
    # Create a visualizer
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window()

    # Add the point cloud to the visualizer
    vis1.add_geometry(pcd)

    # Set the background color to blue (RGB values for blue)
    vis1.get_render_option().background_color = np.asarray([0, 0, 1])  # Blue background

    # Set the view control parameters for a top-down view
    view_control = vis1.get_view_control()
    view_control.set_front([a, b, c])  # Set the front direction
    view_control.set_up([d, e, f])    # Set the up direction

    # Optionally, you can set the zoom level
    view_control.set_zoom(0.7)
    # Render the scene
    vis1.run()

    # Capture the screen image
    vis1.capture_screen_image(alldef_png)

    # Destroy the window to free up resources
    vis1.destroy_window()

    #FOR ROTATED POINT CLOUD
    df = pd.read_csv(rot_path, delimiter=';', header=None)

    # Save the data in XYZ format
    df.to_csv('rotdef.xyz', header=False, index=False, sep=' ')

    # Load the point cloud
    pcd = o3d.io.read_point_cloud("rotdef.xyz")
    os.remove('rotdef.xyz')
    # Get the Z-coordinates (or any other scalar field)
    z_coords = np.asarray(pcd.points)[:, 2]  # Use Z for height-based grayscale

    # Normalize the Z-coordinates to the range [0, 1]
    z_min = z_coords.min()
    z_max = z_coords.max()
    grayscale = (z_coords - z_min) / (z_max - z_min)

    # Convert normalized Z to grayscale colors
    pcd.colors = o3d.utility.Vector3dVector(np.tile(grayscale[:, np.newaxis], (1, 3)))
    
    # Create a visualizer
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window()

    # Add the point cloud to the visualizer
    vis2.add_geometry(pcd)

    # Set the background color to blue (RGB values for blue)
    vis2.get_render_option().background_color = np.asarray([0, 0, 1])  # Blue background

    # Set the view control parameters for a top-down view
    view_control = vis2.get_view_control()
    view_control.set_front([a, b, c])  # Set the front direction
    view_control.set_up([d, e, f])    # Set the up direction

    # Optionally, you can set the zoom level
    view_control.set_zoom(0.7)
    # Render the scene
    vis2.run()


    # Capture the screen image
    vis2.capture_screen_image(rotdef_png)

    # Destroy the window to free up resources
    vis2.destroy_window()

    return ideal_png,alldef_png,rotdef_png


if __name__ == '__main__':
    import cv2
    import os
    current_directory = os.path.dirname(__file__)
    ideal_file_path_pcd=input("Enter file path to your ideal point cloud text file only (.txt): ")
    defected_file_path_pcd=input("Enter path to your ICP_rotated defected point cloud if the orientation of defected and ideal was not same else enter path to your defected point cloud, text file only (.txt): ")
    trudefect_file_path_pcd=input("Enter path to your true defect point cloud text file only (.txt): ")
    choice='yes'
    while (choice=='yes'):
        pcd_view=int(input("FOR RIGHT SIDE VIEW PRESS 1, FOR LEFT SIDE VIEW PRESS 2, FOR TOP VIEW PRESS 3,FOR FRONT VIEW PRESS 4, FOR BACK VIEW PRESS 5"))
        if (pcd_view==1):
            a=1.0
            b=0.0
            c=0.0
            x=0.0
            y=0.0
            z=1.0
        elif (pcd_view==2):
            a=-1.0
            b=0.0
            c=0.0
            x=0.0
            y=0.0
            z=1.0
        elif (pcd_view==3):
            a=0.0
            b=0.0
            c=-1.0
            x=0.0
            y=-1.0
            z=0.0
        elif (pcd_view==4):
            a=0.0
            b=-1.0
            c=0.0
            x=0.0
            y=0.0
            z=1.0
        elif (pcd_view==5):
            a=0.0
            b=1.0
            c=0.0
            x=0.0
            y=0.0
            z=1.0
        

        ideal_file_path,defects_image_path,defected_file_path=Grayscale_capture(ideal_file_path_pcd,defected_file_path_pcd,trudefect_file_path_pcd,a,b,c,x,y,z)
        
        
        check='y'
        frame_number_1 = extract_frame_number(ideal_file_path)
        frame_number_2 = extract_frame_number(defected_file_path)



        # Concatenate into desired format
        comparison = rf"{current_directory}\Results Compared {frame_number_1} with {frame_number_2}"
        

        #Read images
        image1=cv2.imread(ideal_file_path)
        image2=cv2.imread(defected_file_path)

        crop=float(input("Enter percentage crop from both top and bottom that you wish to apply between 0-100\n "))/100
    
        #Preprocesses the image
        image1, image2=image_preprocesing(image1,image2,blur=True,crop=crop)

        temp_file_path1=rf"{current_directory}\temp_ideal_frame{frame_number_1}.png"
        temp_file_path2=rf"{current_directory}\temp_defected_frame{frame_number_2}.png"
        cv2.imwrite(temp_file_path1,image1)
        cv2.imwrite(temp_file_path2,image2)





        #main code used for image alignment
        ideal,defected=main(temp_file_path1,temp_file_path2)






        #max pooling applied to both ideal and defected image
        ideal=max_pooling(ideal)
        defected=max_pooling(defected)



        #ssim comparision b/w ideal image and defected image
        total_no_of_pixels,actual_percentage_similarity=ssim_cal(ideal,defected,save_path=comparison,target_size=[5000,3000],blur=True)
        print(f"SSIM : {actual_percentage_similarity}"  )
        delete_image(temp_file_path1)
        delete_image(temp_file_path2)

        if check.lower()=="y":
            
            image3=cv2.imread(defects_image_path)

        

            image1, image3=image_preprocesing(image1,image3,blur=True,crop=crop)
            
        
            temp_file_path1=rf"{current_directory}\temp1_ideal_frame{frame_number_1}.png"
            temp_file_path2=rf"{current_directory}\temp1_defected_frame{frame_number_2}.png"
            
            
            cv2.imwrite(temp_file_path1,image1)
            cv2.imwrite(temp_file_path2,image3)

            ideal,defects=main(temp_file_path1,temp_file_path2)

            plt.imsave(temp_file_path1,ideal)
            plt.imsave(temp_file_path2,defects)

            it2=cv2.imread(temp_file_path1,cv2.COLOR_RGB2GRAY)
            it3=cv2.imread(temp_file_path2,cv2.COLOR_RGB2GRAY)
            
            
        
            ideal=max_pooling(it2)
            defects=max_pooling(it3)
        
            predicted_pixels=compare_defects_and_ideal(ideal,defects)
            

            delete_image(temp_file_path1)
            delete_image(temp_file_path2)

            accuracy_calc(predicted_pixels,total_no_of_pixels,actual_percentage_similarity)
        choice=input('do u want to test on another view, type yes or no')