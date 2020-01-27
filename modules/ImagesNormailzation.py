import os
import time
import logging
import traceback
import numpy as np
import cv2.cv2 as cv2
from PIL import Image

import FacesMerging as FM
import FaceRcognition as FR
import PoseEstimation as PE
import matplotlib.tri as tri
import matplotlib.pylab as plt

root_File_dir = '/Users/liusiyuan/Desktop/Codes/Img4test/Private'
DRAW_IMAGE_BUTTON = True
IF_STORE_FILE = False

caculation_Times_in_iteration = 0


def Iteration(set_root):
    # All folders and files are found and contained inside variable 'directories_list' as a consequence set.
    dir_list = os.listdir(set_root)
    for current_dir in dir_list:
        # Generate a new path by linking two parts.
        path_temp = os.path.join(set_root, current_dir)
        # Check whether or not new path is directed at a folder.
        if not os.path.isdir(path_temp):
            # If new path is not a folder.
            # Open that file in binary, read the most front four sets of numbers from the file as variable 'serial_number'.
            try:
                # Main Access.
                corresponding_points_set = main_Normalization(path_temp)
                global caculation_Times_in_iteration
                caculation_Times_in_iteration += 1
            except Exception:
                logging.warning('File ' + path_temp +
                                ' cannot be successfully calculated! Error code:305 in ImagesNormailzation.py' + '\n' + traceback.format_exc())
            else:
                if corresponding_points_set == False:
                    logging.warning('File ' + path_temp +
                                    ' cannot be successfully calculated! Error code:306 in ImagesNormailzation.py')
                logging.info('File ' + path_temp + ' has been successfully calculated! Process code:3102 in ImagesNormailzation.py with consequence: ' +
                             '\n' + str(corresponding_points_set))
        else:
            # This suituation means current path is actually a folder, then do the iteration by invoking itself.
            # Promote 'folder_quantity' as a global variable and make addition.
            Iteration(path_temp)


def flip_Horizonally(path):
    flipped_image = cv2.flip(cv2.imread(path), 1)
    if IF_STORE_FILE == True:
        # Split 'current_file_directory' and 'full_name' from 'path_temporary'.
        curr_file_dir, full_name = os.path.split(path)
        # Split 'file_name' and 'extension' from 'full_name'.
        file_name, extension = os.path.splitext(full_name)
        prefix_extension = '_flipped'
        generated_path = (curr_file_dir + '/' + file_name +
                          prefix_extension + extension)
        while not if_Contain(path, prefix_extension):
            if not os.path.exists(generated_path):
                cv2.imwrite(generated_path, flipped_image)
                break
            else:
                logging.warning('Flipped file ' + generated_path +
                                ' already exised! Error code:303 in ImagesNormailzation.py')

    return flipped_image

# if_Contain(file_name, prefix_extension)


def if_Contain(target_str, condition_str):
    target_len = len(target_str)
    condition_len = len(condition_str)
    reversed_list = list(condition_str)
    reversed_list.reverse()
    reversed_str = "".join(reversed_list)
    inner_round_time = 0
    inner_counter = 0
    for current_letter in reversed_str:
        inner_round_time += 1
        if current_letter == target_str[target_len-inner_round_time]:
            inner_counter += 1
            if inner_counter == condition_len:
                return True
    return False


def generate_points_Set(points_set_A, points_set_B):
    pointsArray = []
    pointsArray.append(points_set_A)
    pointsArray.append(points_set_B)
    return pointsArray


def generate_images_Matrix(image_array_A, image_array_B):
    matrixArray = []
    matrixArray.append(np.float32(image_array_A)/255.0)
    matrixArray.append(np.float32(image_array_B)/255.0)
    return matrixArray


def flip_Points_set(target_points_set, width_of_image):
    supposed_points_set = []
    for current_coordinate in target_points_set:
        supposed_points_set.append(
            (int(width_of_image - current_coordinate[0]), int(current_coordinate[1])))
    return supposed_points_set

def seperate_XY(in_set):
    X=[]
    Y=[]
    for current_coordinate in in_set:
        round_time=0
        for current_value in current_coordinate:
            if round_time==1:
                Y.append(current_value)
            if round_time==0:
                X.append(current_value)
                round_time=1
                

    return X,Y

def main_Normalization(path):
    recognised_points_set_A, width_of_image = FR.face_Recognition(path)
    image_points = np.array([
        recognised_points_set_A[30],     # Nose tip
        recognised_points_set_A[8],     # Chin
        recognised_points_set_A[36],     # Left eye left corner
        recognised_points_set_A[45],     # Right eye right corne
        recognised_points_set_A[48],     # Left Mouth corner
        recognised_points_set_A[54]      # Right mouth corner
    ], dtype="double")
    # PE.PoseEstimation(path, image_points)
    if not recognised_points_set_A == False:
        flipped_image = flip_Horizonally(path)
        recognised_points_set_A_flipped, useless_Var = FR.new_face_Recognition(
            flipped_image)
        X,Y=seperate_XY(recognised_points_set_A_flipped)
        print(str(X))
        print(str(Y))
        triangles = tri.Triangulation(X, Y)
        plt.triplot(triangles,'r--')  
        plt.xticks(X,())
        plt.yticks(Y,())
        plt.show()
        # print('recognised_points_set_A_flipped :' + str(recognised_points_set_A_flipped) +
        #       '\n' + 'and width_of_image: ' + str(width_of_image))
        # if DRAW_IMAGE_BUTTON == True:
        #     plt.imshow(flipped_image)
        #     plt.show()
        # Merge both images
        # merged_image = FM.new_merge(generate_points_Set(
        #     recognised_points_set_A, flip_Points_set(recognised_points_set_A, width_of_image)), generate_images_Matrix(cv2.imread(path), flipped_image), width_of_image)
        if bool(recognised_points_set_A_flipped) and bool(width_of_image) == True:
            merged_image = FM.new_merge(generate_points_Set(
                recognised_points_set_A, recognised_points_set_A_flipped), generate_images_Matrix(cv2.imread(path), flipped_image))
            # Normalise the matrix of merged image.
            normalised_image = cv2.normalize(
                merged_image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            #  Show image if request.
            if DRAW_IMAGE_BUTTON == True:
                plt.imshow(normalised_image)
                plt.show()
            recognised_points_set_B = FR.new_face_Recognition(
                normalised_image[:, :, [2, 1, 0]])
            # A function of numpy that shows all the content of a tensor.
            np.set_printoptions(threshold=np.inf)
            logging.info(str(normalised_image[:, :, [2, 1, 0]]))
            logging.info('File ' + path +
                         ' has been successfully merged and recognised. Process code: 3101 in ImagesNormailzation.py')
            if recognised_points_set_B == False:
                logging.warning('File ' + path +
                                ' cannot be recognised after merged! Error code:301 in ImagesNormailzation.py')
            else:
                return recognised_points_set_B
        else:
            logging.warning('File ' + path +
                            ' cannot be recognised after flipped! Error code:304 in ImagesNormailzation.py')
            return False
    else:
        logging.warning('File ' + path +
                        ' cannot be recognised before flipped! Error code:302 in ImagesNormailzation.py')
        return False


def main():
    start_Time = time.time()
    Iteration(root_File_dir)
    end_Time = time.time()
    print('\n' + 'Totally Time use: ' + str(end_Time-start_Time))


if __name__ == '__main__':
    # Initialise log recording function for operation at the first time.
    log_name = ('ImagesNormailzation_log_' + time.strftime('%Y.%m.%d.%H.%M',
                                                           time.localtime(time.time())) + '.log')
    logging.basicConfig(filename=os.path.join(
        os.getcwd()+'/logs', log_name), level=logging.INFO)
    # Print launch time for this program.
    main()
