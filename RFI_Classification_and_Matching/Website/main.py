from flask import Flask, render_template,request
from flask_wtf import FlaskForm
import pandas
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import ocr 
import match_RFI_to_dwg
from waitress import serve
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = '../tmp/'
mode="prod"

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])

def home():
    # sheet_num=request.form['input_Snum']
    form = UploadFileForm()
    #TODO: read the sheet number from the text box and use to return csv file. 
    if form.validate_on_submit():
        #TODO: pass the sheet number from html to flask. 
        file = form.file.data # First grab the file
        if os.path.exists(app.config['UPLOAD_FOLDER']): shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.mkdir(app.config['UPLOAD_FOLDER'])
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        ocr.run() # Extract images from the pdf, extract text from images
        match_RFI_to_dwg.run()
    return render_template('index.html', form=form)

@app.route("/show_data") 
def showData(): # read the excel file from the directory and view in html 
    #TODO: return the csv related to the input sheet number

    df = pandas.read_csv("../tmp/1.csv")
    df.columns=["ID","Title","Question","Answer","Matched Terms"]
    return render_template('index2.html', tables=df,classes="data", titles=[''],header="true") 
#==================================================
if __name__ == '__main__':
    if mode=="dev":
        app.run(host='127.0.0.1', port=5000,debug=True)  
    else:
        serve(app,host='127.0.0.1', port=5000,threads=2)
    #TODO: 
    # Check to see if the results make sense
    # Pick the drawing sheet
    # Modify the program to delete the files
    # improve the speed