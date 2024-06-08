from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow import keras
import cv2 as cv
import numpy as np


class App:
    def __init__(self):
        self.image_path = None
        self.photo = None
        self.model = keras.models.load_model(r'model4.keras')

        self.window = Tk()
        self.window.title('Угадай эмоцию')
        self.window.geometry('600x500')
        self.window.resizable(width=False, height=False)

        self.canvas = Canvas(self.window, width=600, height=500)
        self.canvas.pack()

        self.main_frame = Frame(self.window)
        self.main_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Left frame
        self.left_frame = Frame(self.main_frame)
        self.left_frame.place(relx=0, rely=0, relwidth=0.6, relheight=0.8)

        self.image_label = Label(self.left_frame, text='Photo:', font=40, pady=20)
        self.image_label.pack()

        self.image_preview = Canvas(self.left_frame, width=250, height=250)
        self.image_preview.pack()

        # Right frame
        self.right_frame = Frame(self.main_frame)
        self.right_frame.place(relx=0.6, rely=0, relwidth=0.4, relheight=0.8)

        self.predict_label = Label(self.right_frame, text='Result:', font=40, pady=20)
        self.predict_label.pack()

        self.predict_result = Label(self.right_frame, text='', font=20, pady=10)
        self.predict_result.pack()

        self.emotion0_label = Label(self.right_frame, text='', font=20)
        self.emotion0_label.pack()
        self.emotion1_label = Label(self.right_frame, text='', font=20)
        self.emotion1_label.pack()
        self.emotion2_label = Label(self.right_frame, text='', font=20)
        self.emotion2_label.pack()
        self.emotion3_label = Label(self.right_frame, text='', font=20)
        self.emotion3_label.pack()
        self.emotion4_label = Label(self.right_frame, text='', font=20)
        self.emotion4_label.pack()
        self.emotion5_label = Label(self.right_frame, text='', font=20)
        self.emotion5_label.pack()

        # An example of changing of text
        # predict_result.config(text='New')

        # Bottom frame
        self.bottom_frame = Frame(self.main_frame)
        self.bottom_frame.place(relx=0, rely=0.8, relwidth=1, relheight=0.2)

        self.load_file_btn = Button(self.bottom_frame, text='Choose photo', bg='yellow', command=self.load_image)
        self.load_file_btn.place(relx=0.1, rely=0.4)

        self.clear_btn = Button(self.bottom_frame, text='Clear', bg='yellow', command=self.clear)
        self.clear_btn.place(relx=0.4, rely=0.4)

        self.predict_btn = Button(self.bottom_frame, text='Predict', bg='yellow', command=self.predict)
        self.predict_btn.place(relx=0.7, rely=0.4)

        self.window.mainloop()

    def load_image(self):
        file_path = filedialog.askopenfile(filetypes=[('Images', '*.jpg *.jpeg *.png')])
        if file_path is not None:
            self.clear()
            self.image_path = str(file_path).split("'")[1]
            # self.predict_result.config(text='')
            image = Image.open(self.image_path)
            resized_image = image.resize((250, 250))
            self.photo = ImageTk.PhotoImage(resized_image)

            self.image_preview.create_image(0, 0, anchor='nw', image=self.photo)

    def clear(self):
        self.image_path = None
        self.photo = None
        self.predict_result.config(text='')
        self.emotion0_label.config(text='')
        self.emotion1_label.config(text='')
        self.emotion2_label.config(text='')
        self.emotion3_label.config(text='')
        self.emotion4_label.config(text='')
        self.emotion5_label.config(text='')

    def predict(self):
        if self.image_path is not None:
            image = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, (64, 64))
            image = np.reshape(image, (1, 64, 64, 1))
            image = image / 255.0

            prediction = self.model.predict(image)
            clas = np.argmax(prediction)
            emotions = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            self.predict_result.config(text=emotions[clas])
            self.emotion0_label.config(text=f'Anger: {round(prediction[0][0] * 100, 2)}%')
            self.emotion1_label.config(text=f'Fear: {round(prediction[0][1] * 100, 2)}%')
            self.emotion2_label.config(text=f'Happy: {round(prediction[0][2] * 100, 2)}%')
            self.emotion3_label.config(text=f'Neutral: {round(prediction[0][3] * 100, 2)}%')
            self.emotion4_label.config(text=f'Sad: {round(prediction[0][4] * 100, 2)}%')
            self.emotion5_label.config(text=f'Surprise: {round(prediction[0][5] * 100, 2)}%')
        else:
            print('Choose file')


app = App()
