import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
import re
import time
import os
import csv
from PIL import Image
import pandas as pd
import datetime
import requests
from io import BytesIO
import streamlit as st 
import shutil
import helpers.tesseract as tesseract

class ID_EXTRACT:
    def __init__(self,tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = 'tesseract'
        self.sift = cv2.SIFT_create()
        self.characters_to_remove = ['|', '\n', 'i','i\n']

    def get_processed_image(self,im1,im2):
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect SIFT features and compute descriptors.
        keypoints1, descriptors1 = self.sift.detectAndCompute(im1_gray,None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(im2_gray,None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        good=[]
        for match1,match2 in matches:
            if match1.distance<0.75*match2.distance:
                good.append([match1])
        
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match[0].queryIdx].pt
            points2[i, :] = keypoints2[match[0].trainIdx].pt

        # Find homography
        h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography to warp image
        height, width, _ = im1.shape
        final_image = cv2.warpPerspective(im2, h, (width, height))

        return final_image

    def get_thresh(self,img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

    def img2text(self,img):
        info = pytesseract.image_to_string(img,lang="eng")
        return info

    def roi_from_id(self,im2_reg):
        name_img = im2_reg[320:450,250:]
        sap_id_img = im2_reg[400:500,750:]
        year_img = im2_reg[50:100,850:]
        course_img = im2_reg[440:500,300:750]

        name_th = self.get_thresh(name_img)
        sap_th =  self.get_thresh(sap_id_img)
        year_th = self.get_thresh(year_img)
        course_th = self.get_thresh(course_img)

        return name_th,sap_th,year_th,course_th

    def extract_details(self,name_th,sap_th,year_th,course_th):
        name,sap_id,year,course_str = '',0,0,''

        namee = self.img2text(name_th)
        sap_idd = self.img2text(sap_th)
        yearr = self.img2text(year_th)
        coursee = self.img2text(course_th)

        filtered_characters = filter(lambda x: x not in self.characters_to_remove, namee)
        name = ''.join(filtered_characters)

        for stri in sap_idd.split():
            stri = re.sub(r'[!@#$.,]','',stri)
            if len(stri) == 11: 
                sap_id = stri
        
        for yr in yearr.split():
            if len(yr) == 9:
                year = yr
            
        course_str = ''
        for c in coursee.split():
            course_str+=c
            
        return name,sap_id,year,course_str

    def determine_sr_no(self,file_path): ##3
        reader=csv.reader(open(file_path))
        sr_no=len(list(reader))
        return sr_no

    def choose_folder_dialog(self): ##1
        root = Tk()
        root.withdraw()  # Hide the main window
        folder_path = filedialog.askdirectory(title="Select Folder",master=root)
        return folder_path

    def create_csv_file(self,folder_path,file_name): ##2
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        file_path=folder_path+'/'+ file_name
        
        columns=['Sr No', 'Student Name', 'SAP ID', 'Course', 'Year', 'Attending Date', 'Attending Time','Course Year']

        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as csvfile:
                # Create a CSV writer object
                csv_writer = csv.writer(csvfile)

                # Write the header to the CSV file
                csv_writer.writerow(columns)
        
        return file_path

    def append_into_csv(self,file_path,student_data):
        now=datetime.datetime.now()
        current_date=now.strftime('%x')
        current_time=now.strftime('%X')
        if student_data[4] != 0:
            course_start=int(student_data[4][0:4])
        else:
            course_start = 0
        
        if course_start == 0:
            course_year = 0

        elif int(now.strftime('%m')) >=1 and int(now.strftime('%m'))<7:
            course_year= int(now.strftime('%Y'))-course_start
        else:
            course_year= (int(now.strftime('%Y'))-course_start)+1

        student_data.append(current_date)
        student_data.append(current_time)
        student_data.append(course_year)
        
        with open(file_path, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(student_data)

    def main(self,image,file_path):
        url = 'https://raw.githubusercontent.com/yashoza1203/Id-card-info-extraction/main/ID_info_extraction/static/ID.jpg'
        response = requests.get(url)
        img1 = Image.open(BytesIO(response.content))
        im1 = np.array(img1)
        im2 = np.array(image)

        final_image = self.get_processed_image(im1,im2)
        name_th,sap_th,year_th,course_th = self.roi_from_id(final_image)
        name, sap_id,year,course = self.extract_details(name_th,sap_th,year_th,course_th)
        sr_no = self.determine_sr_no(file_path)

        student_data = [sr_no,name, sap_id, course,year]
        
        self.append_into_csv(file_path,student_data)


start_time = time.time()

import os
tesseract_cmd = os.path.join(os.path.dirname(__file__), "path", "to", "tesseract")
st.sidebar.write(tesseract_cmd)

idd = ID_EXTRACT(tesseract_cmd)

st.title("Extract details from ID")
st.markdown("## By:- Dev, Yash and Rushank 🤘")

file_name  = st.text_input('Enter the File Name where you want to store the details:', 'id_details.csv')
st.write('The current file is', file_name)

query_image = st.file_uploader('Choose an ID',type='.jpg')
submit = st.button('Extract Information')
# st.write(query_image.read())

if submit:
    if query_image is not None:
        # folder_path = idd.choose_folder_dialog()
        folder_path = os.getcwd() + '/static/'
        folder_path = folder_path + '/'
        file_path = idd.create_csv_file(folder_path,file_name)
        csv_path = os.getcwd().replace('\\','/') + '/static/' + file_name
        image = Image.open(query_image)
        new_image = image.resize((512, 512))
        st.sidebar.markdown('ID')
        
        st.sidebar.image(new_image,channels="BGR")
        st.sidebar.write("Filename: ", query_image.name)
        img_path = query_image.name
        img_path = 'static/' + img_path

        idd.main(image,file_path)
        data = pd.read_csv(csv_path,parse_dates=['Attending Date','Attending Time'],index_col=0)
        data.drop(data.filter(regex="Unname"),axis=1, inplace=True)
        st.subheader('Raw data')
        data = data.style.format(thousands='')
        st.dataframe(data ,use_container_width=True)
        st.write("--- %s seconds ---" % (time.time() - start_time))
