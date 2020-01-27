# https://face-recognition.readthedocs.io/en/latest/face_recognition.html#module-face_recognition.api
import face_recognition
import BasicDataCM as bD
from PIL import Image, ImageDraw, ImageFont
import traceback
import logging
import numpy
import time
import sys
import os

# ALLOW or NOT ALLOW files operations inside.
FILE_OPERATION_BUTTON = False
DRAW_IMAGE_BUTTON = True
IF_DEBUG = False
# MAIN ACCESS
# root_File_dir = '/Users/liusiyuan/Desktop/Codes/Img4test'
root_File_dir = '/Users/liusiyuan/Desktop/Codes/Img4test/Private'
single_Face_dir = '/picked_SFP/single_Face'
multiple_Faces_dirm = '/picked_SFP/multiple_Faces'


file_quan, folder_quan = 0, 0

# This function will straightly dive into the deepest layer from the root by iterating itself.


def Iteration(set_root):
    # All folders and files are found and contained inside variable 'directories_list' as a consequence set.
    dir_list = os.listdir(set_root)
    for current_dir in dir_list:
        # Generate a new path by linking two parts.
        path_temp = os.path.join(set_root, current_dir)
        # Check whether or not new path is directed at a folder.
        if not os.path.isdir(path_temp):
            global file_quan
            file_quan += 1
            face_Recognition(path_temp)
        else:
            # This suituation means tcurrent path is actually a folder, then do the iteration by invoking itself.
            # Promote 'folder_quantity' as a global variable and make addition.
            global folder_quan
            folder_quan += 1
            Iteration(path_temp)


perfect_Sample_quan = 0
worse_Sample_quan = 0

# Check how many faces or whether or not does current image have.
# And exert following manipulations.


def new_face_Recognition(image, file_path=None, file_name=None, extension=None, splited_List=None, width_of_image=0):
    points_Set, constraints_Set = [], []
    max_X, max_Y, min_X, min_Y, round_time = 0, 0, 0, 0, 0
    face_landmarks_list = face_recognition.face_landmarks(image)
    # If there is no face data inside the list.
    if str(face_landmarks_list) == '[]':
        logging.warning(str(
            file_path) + ' is recognised as no face inside. Code:203 in FaceRcognition.py')
        return False, False
    # It means it has valid face or faces data inside.
    else:
        face_amount = 0
        # Start dismantling this variable with multiple structures.
        # The numbers of elements inside 'face_landmarks' depend on how many faces dose it have.
        for face_landmarks in face_landmarks_list:
            face_amount += 1
        if face_amount == 1:
            # Whether or not manipulat the file to an appropriate folder.
            if FILE_OPERATION_BUTTON == True:
                bD.manipulate_Files(file_path, single_Face_dir)
            global perfect_Sample_quan
            perfect_Sample_quan += 1
            # temp_round_time = -1
            for facial_feature in face_landmarks.keys():
                # print("The {} in this face has the following {} points: {}".format(
                #     facial_feature, len(face_landmarks[facial_feature]), face_landmarks[facial_feature]))
                # Combine all data into a points set and get the min_X min_Y max_X manx_Y respectively.
                # Get crucial four values to draw four constant functions (a rectangle) that can restrict all facial features inside.
                for temp_coordinate in face_landmarks[facial_feature]:
                    points_Set.append(temp_coordinate)
                    # # Delete repetitive four points(WAY I).
                    # temp_round_time += 1
                    # inner_round_time = 0
                    # whether_deleted = False
                    # for existed_Coordinate in points_Set:
                    #     if temp_coordinate == existed_Coordinate:
                    #         inner_round_time += 1
                    #         if inner_round_time == 2:
                    #             del points_Set[temp_round_time]
                    #             print('temp_round_time: ' +
                    #                   str(temp_round_time) + '\n')
                    #             temp_round_time -= 1
                    #             whether_deleted = True
                    #             break
                    this_X = temp_coordinate[0]
                    this_Y = temp_coordinate[1]
                    # Value max_X nad max_Y at the first interation.
                    if round_time == 0:
                        round_time += 1
                        max_X = min_X = this_X
                        max_Y = min_Y = this_Y
                    if this_X > max_X:
                        max_X = this_X
                    if this_X < min_X:
                        min_X = this_X
                    if this_Y > max_Y:
                        max_Y = this_Y
                    if this_Y < min_Y:
                        min_Y = this_Y
        # If detected face more than 1.
        if face_amount > 1:
            # Whether or not move images have multiple faces to a specific folder.
            if FILE_OPERATION_BUTTON == True:
                bD.manipulate_Files(file_path, multiple_Faces_dirm)
            # Make addition on quantity.
            global worse_Sample_quan
            worse_Sample_quan += 1
            # Record manipulation information.

        # Combine values of as a global points set.
        constraints_Set.append(min_X)
        constraints_Set.append(min_Y)
        constraints_Set.append(max_X)
        constraints_Set.append(max_Y)
        collect_FacialFeature_W2H_ratio(max_X-min_X, max_Y-min_Y)
        # Normalise coordinate set of landmark.
        # Delete repetitive four points(WAY II).
        if len(points_Set) == 72:
            temp_round_time = -1
            while (temp_round_time < 4):
                temp_round_time += 1
                if(temp_round_time == 0):
                    del points_Set[60]
                if(temp_round_time == 1):
                    del points_Set[65]
                if(temp_round_time == 2):
                    del points_Set[65]
                if(temp_round_time == 3):
                    del points_Set[68]
            points_Set = google_Facial_Landmark_normalization(points_Set)
            # print('Image '+str(splited_List[1]) + ':'+'\n')
            # print(str(points_Set) + '\n')
            if DRAW_IMAGE_BUTTON == True:
                draw_Graph(image, points_Set, constraints_Set)
            logging.info(
                'Successfully returned. Process code: 2101 in FaceRcognition.py')
            return points_Set, width_of_image
        else:
            if DRAW_IMAGE_BUTTON == True:
                draw_Graph(image, points_Set, constraints_Set)
            logging.warning(single_Face_dir + '/' + file_name +
                            extension + ' cannot recognised as formal format. Code:202 in FaceRcognition.py')
            return False, False


def face_Recognition(file_path):
    # Split 'current_file_directory' and 'full_name' from 'path_temporary'.
    splited_List = os.path.split(file_path)
    # Split 'file_name' and 'extension' from list.
    file_name, extension = os.path.splitext(splited_List[1])
    # Identify which parameter has the valid value.
    width_of_image = 0
    # Load the jpg file.
    try:
        # Re-writed face_recognition_models.oad_image_file(file, mode='RGB')
        fp = open(file_path, 'rb')
        im = Image.open(fp)
        width_of_image = im.width
        im = im.convert('RGB')
        fp.close()
        # Find all facial features in all the faces in the image.
        # But currently, we only consider the situation with one face inse.
    except Exception:
        logging.warning(single_Face_dir + '/' + file_name +
                        extension + ' has error and cannot be opened. Code:201 in FaceRcognition.py' + '\n' + traceback.format_exc())
        return False
    else:
        return new_face_Recognition(numpy.array(im), file_path, file_name, extension, splited_List, width_of_image)

# Change google landmark sequence into opence landmark sequence.


def google_Facial_Landmark_normalization(points_Set):
    current_Round = -1
    # Never change both sequences.
    current_Seq_List = [56, 61, 62, 63, 64, 61, 64, 68]
    target_Seq_List = [65, 56, 57, 58, 59, 60, 62, 66]
    for current_Seq in current_Seq_List:
        current_Round += 1
        points_Set = swap(points_Set, current_Seq - 1,
                          target_Seq_List[current_Round] - 1)
    return points_Set


def swap(points_Set, current_Seq, target_Seq):
    # print('length: ' + str(len(points_Set))+' current: ' +
    #       str(current_Seq) + ' target: ' + str(target_Seq)+'\n')
    intermediate_Value = points_Set[current_Seq]
    points_Set[current_Seq] = points_Set[target_Seq]
    points_Set[target_Seq] = intermediate_Value
    return points_Set


ff_W2H_ratio = 0


def collect_FacialFeature_W2H_ratio(ff_width, ff_height):
    global ff_W2H_ratio
    if ff_height != 0:
        ff_W2H_ratio = numpy.append(ff_W2H_ratio, ff_width/ff_height)
    # Print current consequence by refreshing.
    # sys.stdout.write('\r' + str(folder_quan) + ' folders contain ' +
    #                  str(file_quan) + ' files have been considered. ' + str(worse_Sample_quan + perfect_Sample_quan) +
    #                  ' pictures have faces ,with ratio: ' + str(round((numpy.mean(ff_W2H_ratio)), 5)) + '.')
    # sys.stdout.flush()


def draw_Graph(image, points_Set, constraints_Set):
    # Create a PIL imagedraw object so we can draw on the picture
    font = ImageFont.truetype(
        '/Users/liusiyuan/Desktop/codes/Moderne_Demi_Oblique.ttf', 13)
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    count_num = 0
    if IF_DEBUG == True:
        for point_temp in points_Set:
            count_num += 1
            d.ellipse((point_temp[0]-2, point_temp[1]-2,
                    point_temp[0]+2, point_temp[1]+2), fill='green')
            d.text(point_temp, str(count_num), 'yellow', font)
        # Draw outline as rectangle.
        d.line([(constraints_Set[0], constraints_Set[1]), (constraints_Set[2], constraints_Set[1]),
                (constraints_Set[2], constraints_Set[3]), (constraints_Set[0], constraints_Set[3]), (constraints_Set[0], constraints_Set[1])], width=1, fill='crimson')
    # point_Set_temp = readPoints(
    #     '')
    # count_num = 0
    # for point_temp in point_Set_temp:
    #     count_num += 1
    #     d.ellipse((point_temp[0]-2, point_temp[1]-2,
    #                point_temp[0]+2, point_temp[1]+2), fill='blue')
    #     d.text(point_temp, str(count_num), 'red', font)

    # Show the picture
    pil_image.show()


def readPoints(path):
    # Create an array of points.
    points = []
    # Read points from filePath
    with open(path) as file:
        for line in file:
            temp_round_time = 0
            x, y = 0, 0
            list_From_line = line.split(' ')
            for coordinate in list_From_line:
                temp_round_time += 1
                residue = temp_round_time % 2
                while residue == 1:
                    x = coordinate
                    break
                while residue == 0:
                    y = coordinate
                    points.append((int(x), int(y)))
                    break
    return points


def main():
    Iteration(root_File_dir)
    # Print the final consequence.
    # print('\n' + 'Among them, ' +
    #       str(perfect_Sample_quan) + ' pictures are perfect samples, and ' + str(worse_Sample_quan) +
    #       ' are not. ' + '\n' + str(((worse_Sample_quan + perfect_Sample_quan)/file_quan)*100) +
    #       ' percent of files have face or faces inside that can be recognised.')


if __name__ == '__main__':
    # Initialise log recording function for operation in this time.
    log_name = ('FaceRcognition_log_' + time.strftime('%Y.%m.%d.%H.%M',
                                                      time.localtime(time.time())) + '.log')
    logging.basicConfig(filename=os.path.join(
        os.getcwd()+'/logs', log_name), level=logging.INFO)
    logging.info(time.strftime('%Y.%m.%d.%H.%M',
                               time.localtime(time.time()))+'\n')
    # Print launch time for this program.
    start_Time = time.time()
    main()
    end_Time = time.time()
    print('\n' + 'Time use: ' + str(end_Time-start_Time))
