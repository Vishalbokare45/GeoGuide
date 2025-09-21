#  Author: Rajas Bhosale
#  Filename: Detection.py
#  Functions: search_char, del_val, format_print, classify_event, crop, cropping_v1, character, recognize_characters, filter_contours, task_4a_return, detect
#  Global Variables: classes, reader, val_lis



from sys import platform
import numpy as np
import easyocr
import subprocess
import cv2 as cv       # OpenCV Library
import shutil
import ast
import sys
import os  
import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision.models as models
import cv2   
import time      

#classes to be detected
classes=['combat', 'destroyed_buildings', 'fire', 'humanitarian_aid', 'military_vehicles']

reader = easyocr.Reader(['en'])

val_lis=[False,False,False,False,False] #to get to know how many characters are recognised

def search_char(i,val):
    
    #  * Function Name: search_char
    #  * Input:
    #     - i (str): The character to search for.
    #     - val (str): The string in which to search for the character.
    #  * Output: 
    #     This function does not return any value. It updates a global list 'val_lis'.
    #  * Logic: 
    #     This function searches for a specific character 'i' within a given string 'val' and updates 
    #     a global list 'val_lis' based on the search result. If the character 'i' is found in the string 'val',
    #     or if certain alternative characters are found in 'val', the corresponding index in 'val_lis' is set to True.
    #     Otherwise, the index remains False. To minimise the error in the charcater recognition.
    #  * Example Call: 
    #     search_char('A', 'A4')
        
    # Example scenario:
    
    # Let's assume 'val' is the string 'AC4' and we want to search for the character 'A'.
    # - The function checks if 'A' is present in 'val' or if '4' is present (alternative for 'A').
    # - If either of these conditions is true, it sets the first element of 'val_lis' to True.
    # - Otherwise, it remains False.
    
    
    if i=='A':
        if('A' in val ) or ('4' in val):
            val_lis[0]=True
    if i=='B':
        if('B' in val ) or ('Q' in val) or ('R' in val):
            val_lis[1]=True
    if i=='C':
        if('C' in val ) or ('O' in val):
            val_lis[2]=True 
    if i=='D':
        if('D' in val ) or ('O' in val):
            val_lis[3]=True     
    if i=='E':
        if('F' in val ) or ('E' in val):
            val_lis[4]=True                
            

def del_val(dic):
    
    #  * Function Name: del_val
    #  * Input:
    #     - dic (dict): The dictionary from which values are to be deleted based on conditions.
    #  * Output: 
    #     - dic (dict): The modified dictionary after deleting values based on specified conditions.
    #  * Logic: 
    #     This function takes a dictionary 'dic' as input and deletes values based on conditions defined 
    #     by the global list 'val_lis'. For each character, if the corresponding index in 'val_lis' is True,
    #     the function deletes the value associated with that character key from the dictionary.
    #  * Example Call: 
    #     dictionary = {'A': 'fire', 'B': 'combat', 'C': 'military_vehicles'}
    #     updated_dictionary = del_val(dictionary)
    #     print(updated_dictionary)
        
    
    if(val_lis[0]):
        dic['A']=""
    if(val_lis[1]):
        dic['B']=""
    if(val_lis[2]):
        dic['C']=""
    if(val_lis[3]):
        dic['D']=""
    if(val_lis[4]):
        dic['E']=""
    return dic
        
def format_print(df):
    # """
    #  * Function Name: format_print
    #  * Input:
    #     - df (dict): A dictionary containing key-value pairs to be formatted and printed.
    #  * Output: 
    #     This function does not return any value. It prints the formatted dictionary.
    #  * Logic: 
    #     This function iterates through the key-value pairs of the input dictionary 'df' and formats 
    #     the values based on specified conditions. If a value matches certain strings, it replaces 
    #     the value with a corresponding formatted string in a new dictionary 'new_dic'. The function 
    #     then prints the formatted dictionary as a single-line string enclosed in curly braces.
    #  * Example Call: 
    #     dictionary = {'A': 'fire', 'B': 'combat', 'D': 'humanitarian_aid'}
    #     format_print(dictionary)
    # """
    
    new_dic={}
    for key,val in df.items():
        if(val=="destroyed_buildings"):
            new_dic[key]="Destroyed buildings"
        if(val=="humanitarian_aid"):
            new_dic[key]="Humanitarian Aid and rehabilitation"
        if(val=="military_vehicles"):
            new_dic[key]="Military Vehicles"
        if(val=="fire"):
            new_dic[key]="Fire"
        if(val=="combat"):
            new_dic[key]="Combat"
        output = '{'
        for key, value in new_dic.items():
            output += key + ': ' + value + ', '
        output = output.rstrip(', ') + '}'
    print(output)

def classify_event(image):
    # """
    #  * Function Name: classify_event
    #  * Input:
    #     - image (str): The file path of the image to be classified.
    #  * Output: 
    #     - event (str): The predicted event/class of the input image.
    #  * Logic: 
    #     This function loads a pre-trained ResNet18 model from a checkpoint file, replaces its final fully connected layer
    #     with a new linear layer with the appropriate number of output features, and loads the model's weights from the checkpoint.
    #     It then prepares the input image by applying a series of transformations such as resizing, center cropping, converting to tensor, 
    #     and normalization. The transformed image is passed through the model to obtain the output logits.
    #     The index corresponding to the maximum logit value is used to determine the predicted event/class, which is then returned.
    #  * Example Call: 
    #     predicted_event = classify_event('path/to/image.jpg')
    # """
    
    checkpoint=torch.load('trained_model14.pth') # loading the check point
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(checkpoint)
    model.eval() #set the model to eval mode
    
    #set image transform. 
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image=Image.open(image) #opening image
    
    image_tensor=transformer(image).float()#transforming it
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input=torch.autograd.Variable(image_tensor)

    output=model(input)#output from the model
    
    index=output.data.numpy().argmax()
    event = classes[index]
    return event

def crop(cord,image):
    # """
    #  * Function Name: crop
    #  * Input:
    #     - cord (list): A list containing the coordinates of two points defining a rectangle.
    #       The points should be in the format [x1, y1, x2, y2], where (x1, y1) and (x2, y2) are the top-left and bottom-right corners of the rectangle respectively.
    #     - image (numpy.ndarray): An input image represented as a NumPy array.
    #  * Output: 
    #     - cropped_image (numpy.ndarray): The cropped region of the input image, resized to a square of size 256x256.
    #  * Logic: 
    #     This function takes a rectangle defined by two points and crops the input image to this rectangular region.
    #     It first calculates the center coordinates of the rectangle and the side length of the square to be cropped.
    #     Then, it computes the coordinates of the top-left and bottom-right corners of the square.
    #     Using these coordinates, it crops the image to the square region and resizes it to a square of size 256x256.
    #  * Example Call: 
    #     cropped = crop([100, 50, 300, 250], input_image)
    # """
    
    x1=cord[0]                 # x-coordinate of top-left corner of rectangle
    y1=cord[1]                 # y-coordinate of top-left corner of rectangle
    x2=cord[2]                 # x-coordinate of bottom-right corner of rectangle
    y2=cord[3]                 # y-coordinate of bottom-right corner of rectangle
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    side_length = max(abs(x1 - x2), abs(y1 - y2))

    # Calculate the coordinates of the square's corners
    top_left_x = center_x - side_length // 2
    top_left_y = center_y - side_length // 2
    bottom_right_x = center_x + side_length // 2
    bottom_right_y = center_y + side_length // 2

    # Crop the image to the square
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    cropped_image=cv2.resize(cropped_image, (256, 256))
    # Display or save the cropped image
    return cropped_image

def cropping_v1(frame):
    # """
    #  * Function Name: cropping_v1
    #  * Input:
    #     - frame (numpy.ndarray): An input frame/image represented as a NumPy array.
    #  * Output: 
    #     - cropped_image (numpy.ndarray): The cropped region of the input frame/image.
    #  * Logic: 
    #     This function performs cropping on an input frame/image of the arena using manually defined corner points.
    #     It defines four corner points (x1, y1), (x2, y2), (x3, y3), and (x4, y4) to form a rectangle.
    #     The bounding rectangle around these points is calculated, which defines the region of the arena to be cropped.
    #     Finally, this cropped arena is returned.
    #  * Example Call: 
    #     cropped = cropping_v1(input_frame)
    # """
    
    x1, y1 = 400,40  # top-left
    x2, y2 = 1361,40 # top-right
    x3, y3 = 1369,1030  # bottom-right
    x4, y4 = 400,1030  # bottom-left

    # Define the rectangle to crop
    pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    rect = cv2.boundingRect(np.array(pts))

    # Get the coordinates for cropping
    x, y, w, h = rect

    # Crop the image
    cropped_image = frame[y:y + h, x:x + w]
    return cropped_image


def character(frame):
    # """
    #  * Function Name: character
    #  * Input:
    #     - frame (numpy.ndarray): An input frame/image represented as a NumPy array.
    #  * Output: 
    #     - text (str): Recognized characters extracted from the input frame/image.
    #  * Logic: 
    #     This function performs character recognition on an input frame/image. 
    #     It first converts the frame to grayscale and applies thresholding to obtain a binary image.
    #     Contours are then found in the binary image and filtered based on their area.
    #     The filtered contours are drawn on the original frame, and for each contour, the function extracts 
    #     the region of interest (ROI) and performs character recognition using a specified method.
    #     The recognized characters are concatenated together to form a text string, which is returned as the output.
    #  * Example Call: 
    #     text = character(input_frame)
    # """
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    filtered_contours = filter_contours(contours, min_area=500)

    # Draw contours on the original frame
    cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 2)
    # Process each contour region
    text=""
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Extract the region of interest (ROI)
        roi = frame[y:y+h, x:x+w]

        # Recognize characters using EasyOCR
        text+= recognize_characters(roi, reader)
        
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    return text

def recognize_characters(roi, reader):
    # """
    #  * Function Name: recognize_characters
    #  * Input:
    #     - roi (numpy.ndarray): The region of interest (ROI) containing characters to be recognized, represented as a NumPy array.
    #     - reader: An object or function responsible for recognizing characters within the ROI.
    #  * Output: 
    #     - recognized_text (str): Recognized characters extracted from the ROI.
    #  * Logic: 
    #     This function performs character recognition on a region of interest (ROI) using a specified reader.
    #     It takes the ROI and the reader object/function as input. The reader is assumed to be capable of recognizing
    #     characters within the ROI and returning the results.
    #     The function invokes the reader to recognize characters within the ROI and retrieves the recognition results.
    #     It then extracts the recognized text from the results and concatenates it into a single string, which is returned as output.
    #  * Example Call: 
    #     recognized_text = recognize_characters(roi_image, ocr_reader)
    # """
    
    results = reader.readtext(roi)
    return " ".join([result[1] for result in results])

def filter_contours(contours, min_area):
    # """
    #  * Function Name: filter_contours
    #  * Input:
    #     - contours (list): A list of contours, where each contour is represented as a numpy.ndarray.
    #     - min_area (int): The minimum area threshold used to filter contours. Contours with an area greater than this threshold are retained.
    #  * Output: 
    #     - filtered_contours (list): A filtered list of contours that satisfy the area threshold condition.
    #  * Logic: 
    #     This function filters contours based on their area using a specified minimum area threshold.
    #     It iterates over the input list of contours and retains only those contours whose area is greater than the specified minimum area threshold.
    #     The area of each contour is calculated using the OpenCV function cv2.contourArea().
    #     The contours that meet the area threshold condition are stored in a new list, which is returned as the output.
    #  * Example Call: 
    #     filtered_contours = filter_contours(contours_list, min_area_threshold)
    # """
    
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def task_4a_return():
    # """
    #  * Function Name: task_4a_return
    #  * Input: None
    #  * Output: None
    #  * Logic:
    #     This function initializes an empty dictionary named 'identified_labels'.
    #     It does not take any input parameters.
    #     The purpose of this function seems to be to prepare a data structure to store identified labels or results for task 4a.
    #     However, as the function body is empty and there are no further instructions, it does not perform any specific operations.
    #     It simply creates an empty dictionary but does not utilize or modify it.
    #  * Example Call: 
    #     task_4a_return()
    # """
    
    identified_labels = {}  
    
    #mapping the images from arena
    mapping = {
    'E': (193,79,305,182),
    'D': (173,415,276,530),
    'C': (682,425,790,530),
    'B': (682,632,780,740),
    'A': (184,860,275,960),
    }
    #mapping of charcters from the arena
    charc={
    'E': (193,79,305,182),
    'D': (173,415,276,530),
    'C': (682,425,790,530),
    'B': (682,632,780,740),
    'A': (184,860,275,960),
    }
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!
    color = (0, 255, 0)  # Define the color in BGR format (here, it's green)
    thickness = 2  # Define the thickness of the rectangle border
    #extract the highest resolution image
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    n=20# variable to remove out empty frames
    curr=time.time() #to get the current time 
    print("Wait for 15 sec to detect")
    while True:
        ret, frame1 = cap.read()
        # to pass the empty frames
        while(n)>0:
            ret, frame1 = cap.read()
            n-=1
        #arena cropping
        frame=cropping_v1(frame1)
        #rotating the image
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for i in ['A','B','C','D','E']:
            cropped_image=crop(mapping[i],frame)  # to take out each image frame
            cv2.rectangle(frame, (mapping[i][0],mapping[i][1]), (mapping[i][2],mapping[i][3]), color, thickness)
            file_name="img_"+i+".png"
            cv2.imwrite(file_name, cropped_image) #saving the file to specific name
            char_crop=crop(charc[i],frame) #cropping charcater
            val=character(char_crop)
            search_char(i,val)
            class_name=classify_event(file_name)
            text_position = (mapping[i][0] - 10, mapping[i][1])  # Adjust the x-coordinate for text placement
            cv2.putText(frame, class_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            identified_labels[i]=class_name       
        cv2.imshow('Live Camera', frame)
        if(time.time()-curr>20):
            break
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()        # Releases the capture device (camera).
    cv2.destroyAllWindows()       # Destroys all OpenCV windows.
    identified_labels=del_val(identified_labels)    # Removes any empty values from the identified_labels dictionary.
    print(val_lis)
    final_val={}
    for key in identified_labels.keys():          # Iterates through the keys of identified_labels dictionary.
        if(identified_labels[key]!=""):           # Checks if the value associated with the key is not an empty string.
            final_val[key]=identified_labels[key] # Assigns the value to the key in the final_val dictionary.

    return final_val



def detect():
    # """
    #  * Function Name: detect
    #  * Input: None
    #  * Output: 
    #     - identified_labels (dict): A dictionary containing identified labels.
    #  * Logic:
    #     This function serves as a part of a detection process.
    #     It first calls the function task_4a_return() to obtain a dictionary named 'identified_labels',
    #     which presumably contains information about detected objects or labels from a previous task.
    #     Next, it calls the function format_print() to print or format the contents of the 'identified_labels' dictionary.
    #     Finally, it returns the 'identified_labels' dictionary.
    #  * Example Call: 
    #     detected_labels = detect()
    # """
    
    identified_labels = task_4a_return()
    format_print(identified_labels)
    return identified_labels
