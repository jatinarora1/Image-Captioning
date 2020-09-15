from flask import Flask, url_for, request, render_template, jsonify, flash, redirect
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
import os
import numpy as np
import images as Caption_it

# __name__ == __main__
app = Flask(__name__)

@app.route('/', methods= ['GET', 'POST'])
def main():
	if request.method == 'GET':
        return(render_template('main.html'))
    if request.method == 'POST':

		f = request.files['userfile']
		path = "./static/{}".format(f.filename)# ./static/images.jpg
		f.save(path)

		caption = Caption_it.caption_this_image(path)
		
		result_dic = {
		'image' : path,
		'caption' : caption
		}

	return render_template("main.html", your_result =result_dic)

if __name__ == '__main__':
	# app.debug = True
	app.run()