# Abusive Text Classifier

A client-server approach to censoring abusive text in your web browser.



## How it works

***

The approach is made up of 2 components:

1. Browser extension

2. Server



For the system to work, the server application must be running and the browser extension must be enabled. The user can then simply visit any webpage to see *most* abusive text censored (the current model is able to correctly classify 96% of non-abusive samples and 69% of abusive samples*). 

1. The browser extension works by finding each text section of a loaded webpage and sending them to the server. If the server responds saying the section contains negative text, that section will be censored. 

2. The server's main function is listening for requests with text data. It takes the text from each request and uses a pre-trained machine learning model to predict whether it is abusive or not. This prediction result is then sent back as a response to the request. 

*On a test dataset sourced from Wikipedia forums which was separate from the training data. 

## Instructions

---

Core requirements:

- Python 3.9.13

- Chromium browser (i.e., Google Chrome) 

Optional requirement:

- Git

- Pip



The first step is to download the project folder to your local machine. This can be done by clicking the "Code" button on this page and then "Download ZIP". If you have Git installed, you can alternatively open a terminal in your directory of choice and enter:

> git clone https://github.com/AA760/abusive-text-classifier.git



Next, you will need to install all the necessary Python dependencies. The easiest way to do this is by using pip, which should be installed by default with Python. Simply open up a terminal in the root of the project folder you downloaded and enter:

> pip install -r requirements.txt



Then, in the same terminal or a new one, navigate to the ****abusive-text-classifier\server**** directory and run the server application with the following command:

> python server.py



Now, launch your Chromium browser extension and navigate to the extensions page. This can be done by entering the following into the address bar:

> chrome://extensions/



Ensure the "Developer Mode" toggle in the top right is on. Then click the "Load unpacked" button. Navigate to the ****abusive-text-classifier\chrome-extension****

directory and click "Select Folder". 



A new extension card will appear named "Abusive Text Classifier". Ensure that the bottom-right toggle is set to on. 



You should now be able to visit any webpage as normal and see abusive text be censored in real-time. By default, the server is set to debug mode and you will be able to see text logs in the server terminal if you are interested. 
