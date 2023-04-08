## Request for Information (RFI) Recommender System for Pre-Construction Design Review Application Using Natural Language Processing, Chat-GPT, and Computer Vision 

(1) Roshan Panahi, (2) John-Paul Kivlin, (3) Joseph Louis 

(1) Ph.D. Candidate, School of Civil and Construction Engineering, Oregon State University, Corvallis Oregon 97331 USA, E-mail: panahir@oregonstate.edu <br>
(2) Senior Project Manager, Skanska, Portland, Oregon 97209 USA, E-mail: johnpaul.kivlin@skanska.com <br>
(3) Assistant Professor, Dept. of Civil and Construction Engineering, Oregon State University, Corvallis Oregon 97331 USA, E-mail: joseph.louis@oregonstate.edu <br>

## Abstract:
Design reviews are critical for construction projects to reduce costly reworks and future conflicts. However, this is a challenging task due to uncertainties during the initial stages of a project which can lead to numerous Requests for Information (RFIs). With the recent advancements in language models and computer vision, a large volume of historical RFIs can be leveraged to aid design reviews. This study proposes a novel framework using natural language processing, ChatGPT API, and computer vision techniques to identify the RFIs from previous projects that are more likely to reoccur in the project under review. The framework was tested using RFI data from 19 healthcare construction projects, and a web application was used to evaluate user experiences with the tool. Successful implementation of the proposed framework could reduce the number of RFIs, change orders, rework by contractors, and the likelihood of time and cost overruns for construction projects. 
Keywords: design review, request for information (RFI), natural language processing, computer vision, Chat-GPT

## Installation:
The installation instructions are for Ubuntu. 

1. Clone "RFI_Classification_and_Matching" repository using: "git clone https://github.com/RoshanPanahi2019/Datascience/tree/main/RFI_Classification_and_Matching"
2. Create an anaconda environment using: "conda create --name myenv"
3. Activate the conda environment using: "conda activate myenv"
4. Install pip inside the new environment: "conda install pip"
5. Install all packages using pip: "pip install -r requirements.txt"
6. Execute the script using: "python main.py"
7. Open a browser and type: "http://localhost:5000/"
8. Click on "browse", select a drawing.pdf, and press "upload.
9. After the process if over, click on "Show RFI" 
