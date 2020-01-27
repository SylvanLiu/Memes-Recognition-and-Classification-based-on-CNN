# encoding:utf-8
import os
import sys
import time
import numpy
import logging
import traceback
from PIL import Image

# Ignore annoying futherWarning and userWarning.
import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib
    import seaborn

# This should be launch first.

# ALLOW or NOT ALLOW files operations inside.
FILE_OPERATION_BUTTON = False
# MAIN ACCESS
root_File_dir = '/unchanged_Dataset/Image1'

# Max width and hight constraints to images you want.
MAX_WIDTH, MAX_HTIGHT = 1080, 1080

# Other functional settings.
dynamic_Img_dir = '/abandoned/Dynamic'
overly_Big_dir = '/abandoned/Overly_big'
irrelevant_Img_dir = '/abandoned/Irrelevant'
recycle_Bin_dir = '/Users/liusiyuan/.Trash'

# Declare and initialise global variables.
file_quan, folder_quan = 0, 0
x_Set, y_Set, ratio_Set = 0, 0, 0

# Ceate a global dictionary with four default conditions.
noteworthy_Serialnumber_list = {'[255, 216, 255, 224]': 0,
                                '[137, 80, 78, 71]': 0, '[255, 216, 255, 225]': 0, '[71, 73, 70, 56]': 0}


def manipulate_Files(path_temp, new_dir='', new_type=''):
    try:
        # Split 'current_file_directory' and 'full_name' from 'path_temporary'.
        curr_file_dir, full_name = os.path.split(path_temp)
        # Split 'file_name' and 'extension' from 'full_name'.
        file_name, extension = os.path.splitext(full_name)
        # Identify which parameter has the valid value.
        if int(len(new_dir)) != 0:
            curr_file_dir = new_dir
        if int(len(new_type)) != 0:
            extension = new_type
        # Generate new path.
        new_path = curr_file_dir + '/' + file_name + extension
        # Main manipulation.
        os.rename(path_temp, new_path)
        # If file with right extension already exists in the current directory.
    except FileExistsError:
        # Remove repetitive files to recycle bin, and it allows to move the files inside with exited names.
        manipulate_Files(path_temp, recycle_Bin_dir)
    except Exception:
        # Record error.
        logging.warning(
            'File ' + path_temp + ' renamed Unsuccessfully! Error code:104 in BasicDataCM.py' + '\n' + traceback.format_exc())
    else:
        # Record manipulation.
        logging.info('File: ' + new_path +
                     ' has been successfully processed. Process code:1105')


# Fix files with wrong extensions and return a new path for images information collection.
def format_Correction(path_temp, serial_num):
    # Split 'current_file_directory' and 'full_name' from 'path_temporary'.
    curr_file_dir, full_name = os.path.split(path_temp)
    # Split 'file_name' and 'extension' from 'full_name'.
    file_name, extension = os.path.splitext(full_name)
    supposed_type = ''
    if serial_num == [71, 73, 70, 56]:
        supposed_type = '.gif'
        manipulate_Files(path_temp, dynamic_Img_dir, supposed_type)
        return (dynamic_Img_dir + '/' + file_name + supposed_type)
    else:
        if serial_num == [255, 216, 255, 224] or [255, 216, 255, 225]:
            supposed_type = '.jpg'
        if serial_num == [137, 80, 78, 71]:
            supposed_type = '.png'
        if supposed_type != extension:
            manipulate_Files(path_temp, curr_file_dir, supposed_type)
    # Return the new_path instead of continuing using the old one.
        return (curr_file_dir + '/' + file_name + supposed_type)


# Collect statistical data from images, put their width and height into 'x_Set' and 'y_Set' co-ordinates sets.


def collect_Img_resolutions(path_temp):
    global file_quan
    file_quan += 1
    # Opening imgaes in this kind of way is to avoid files being occupied.
    # Because library 'Image' doesn`t provide a function of closing opened images.
    try:
        curr_file = open(path_temp, 'rb')
        curr_image = Image.open(curr_file)
        # Get their width and height.
        img_width = curr_image.width
        img_height = curr_image.height
        # Close images timely for the following files operations.
        curr_file.close()
        # Identify whether or not current file meets our needs in resolution.
        if FILE_OPERATION_BUTTON == True:
            global MAX_WIDTH, MAX_HTIGHT
            if not (img_width <= MAX_WIDTH and img_height <= MAX_HTIGHT):
                manipulate_Files(path_temp, overly_Big_dir)
    except Exception:
        # Record error.
        logging.warning('File ' + path_temp +
                        ' manipulated Unsuccessfully! Error code:106 in BasicDataCM.py' + '\n' + traceback.format_exc())
    # Collect statistical data if no errors happened.
    else:
        global x_Set, y_Set, ratio_Set
        x_Set = numpy.append(x_Set, img_width)
        y_Set = numpy.append(y_Set, img_height)
        ratio_Set = numpy.append(ratio_Set, img_width/img_height)
    # Fresh console by new notification.
    sys.stdout.write('\r' + str(folder_quan) + ' folders contain ' +
                     str(file_quan) + ' files have been considered.')
    sys.stdout.flush()

# To collect that how many kinds of serial numbers are contained by the universal set and return a dictionary.


def collect_Format(path_temp, serial_num):
    global noteworthy_Serialnumber_list
    # If true, the file belongs to basic four types ,make counting and manipulations.
    if serial_num == [255, 216, 255, 224] or serial_num == [137, 80, 78, 71] or serial_num == [255, 216, 255, 225] or serial_num == [71, 73, 70, 56]:
        noteworthy_Serialnumber_list[format(serial_num)] = format(
            int(noteworthy_Serialnumber_list[format(serial_num)])+1)
        # File operation access.
        if FILE_OPERATION_BUTTON == True:
            # Collect valid info of current image with format correction.
            collect_Img_resolutions(
                format_Correction(path_temp, serial_num))
        else:
            # Collect valid info of current image directly.
            collect_Img_resolutions(path_temp)
    else:
        # File operation access.
        # Abandon useless images.
        if FILE_OPERATION_BUTTON == True:
            manipulate_Files(path_temp, irrelevant_Img_dir)
        # Collect new formats and make counting for formats already existed.
        temp_round_time_l = 0
        for noteworthy_Serialnumber in noteworthy_Serialnumber_list:
            # If  true, it manifests this is a type already existed, so plus one on its amount.
            if str(serial_num) == noteworthy_Serialnumber:
                noteworthy_Serialnumber_list[format(serial_num)] = format(
                    int(noteworthy_Serialnumber_list[format(serial_num)])+1)
                break
            # If not true, try to count and find out whether or not it is a whole new type.
            if str(serial_num) != noteworthy_Serialnumber:
                temp_round_time_l += 1
                # If true, it manifests this is a whole new type, so create it in the dictionary.
                if temp_round_time_l == len(noteworthy_Serialnumber_list):
                    noteworthy_Serialnumber_list[format(serial_num)] = 1

# This function will straightly dive into the deepest layer from the root by iterating itself.


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
                file = open(path_temp, 'rb')
                turn_num_while = 0
                serial_num = []
                while turn_num_while < 4:
                    temp_content = file.read(1)
                    turn_num_while += 1
                    # Convert numbers into decimal forms then store into a list.
                    serial_num.append(ord(temp_content))
                # Close files timely to avoid being occupied when we manipulate them.
                file.close()
                # ONLY ACCESS.
                collect_Format(path_temp, serial_num)
            except Exception:
                logging.warning('File ' + path_temp +
                                ' is broken! Error code:108 in BasicDataCM.py' + '\n' + traceback.format_exc())
        else:
            # This suituation means current path is actually a folder, then do the iteration by invoking itself.
            # Promote 'folder_quantity' as a global variable and make addition.
            global folder_quan
            folder_quan += 1
            Iteration(path_temp)


def draw_Graphs():

    print('Average of widths： ' + str(numpy.mean(x_Set)))
    print('Average of heights： ' + str(numpy.mean(y_Set)))
    print('Average of widths to heights ratio： ' + str(numpy.mean(ratio_Set)))
    print('Serial numbers set: ' + str(noteworthy_Serialnumber_list) + ' with ' +
          str(len(noteworthy_Serialnumber_list)) + ' situations. ')

    seaborn.jointplot(x_Set, y_Set, kind="reg", color="#FC9D9A", xlim=(0, 1200), ylim=(0, 1200),
                      space=0, height=8, ratio=16)
    # matplotlib.pyplot.savefig('1.png', format='png',
    #                           transparent=True, dpi=2000, pad_inches=0)

    seaborn.jointplot(x_Set, y_Set, kind="hex",
                      color="#FC9D9A", xlim=(0, 1200), ylim=(0, 1200))
    # matplotlib.pyplot.savefig('2.png', format='png',
    #                               transparent=True, dpi=2000, pad_inches=0)
    matplotlib.pyplot.show()

# Main


def main():
    try:
        # Making erogodicity from the root path as a variable in the brackets.
        # Using regular expression in order to correct path to fit the path rules on OSX.
        # Main entrance of this python.
        # Change the only entrance path here.
        Iteration(root_File_dir)
    except Exception:
        sys.stdout.flush()
        print('\n' + '_______________________Calculation Partly Completed_______________________')
        # Print out errors.
        logging.warning('Calculation Partly Completed.' +
                        '\n' + traceback.format_exc())
        draw_Graphs()
    else:
        sys.stdout.flush()
        print('\n' + '______________________Calculation Entirely Completed______________________')
        draw_Graphs()


if __name__ == '__main__':
    # Initialise log recording function for operation at the first time.
    log_name = ('BasicDataCM_log_' + time.strftime('%Y.%m.%d.%H.%M',
                                                   time.localtime(time.time())) + '.log')
    logging.basicConfig(filename=os.path.join(
        os.getcwd()+'/logs', log_name), level=logging.INFO)
    # Record time.
    logging.info(time.strftime('%Y.%m.%d.%H.%M',
                               time.localtime(time.time()))+'\n')
    # Print launch time for this program.
    start_Time = time.time()
    main()
    end_Time = time.time()
    print('\n' + 'Time use: ' + str(end_Time-start_Time))
