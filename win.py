import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cc.Core import *

captcha_name = "supreme_court"
argv = sys.argv
exec = os.path.basename(argv[0])
base_dir = os.path.abspath(os.path.dirname(__file__))
meipass = getattr(sys, '_MEIPASS', None)
if meipass:
    base_dir = meipass
images_dir = os.path.join(base_dir, "images")
model_dir = os.path.join(base_dir, "model")
train_data:TrainInfo = {}
model:Model = {}

window = None
selected_captcha = None
btnFileBrowser = None
txtImage = None
btnExec = None
canvas = None
txtPred = None

def init():
    global train_data, model, window, btnFileBrowser, txtImage, btnExec, canvas, txtPred
    train_data = TrainInfo(base_dir=images_dir, model_path=model_dir)
    model = Model(train_data=train_data, weights_only=True, verbose=0)
    model.load_prediction_model()
    pred_data_list = train_data.get_data_files(False)

    if pred_data_list != None and len(pred_data_list) > 0:
        pred_image = pred_data_list[0]
        model.predict(pred_image)

    return

def predict_captcha(image_path: str):
    temp_dir = os.path.abspath("./temp")

    if os.path.exists(temp_dir) == False:
        os.makedirs(temp_dir)

    temp_image_path=os.path.join("temp", f"{time.time():12.0f}.png")
    
    with Image.open(image_path) as image:

        if image.mode in ('RGBA', 'LA'):
            background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
            background.paste(image, image.split()[-1]) # omit transparency
            image = background

        image.save(temp_image_path)
        
    pred = model.predict(temp_image_path)
    os.remove(temp_image_path)
    return pred

def cli():
    image_path = argv[1]
    pred = predict_captcha(image_path)
    sys.stdout.write(pred)

def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("PNG files", "*.png"), ("All files", "*.*"))
    )

    if file_path:
        execute_pred(file_path)
    return

def execute_pred(image_path: str):
    canvas.delete("all")
    txtPred.delete(0, 'end')
    is_valid = os.path.exists(image_path) and os.path.isfile(image_path)

    if is_valid:
        set_path(image_path)
        set_image(image_path)
        pred = predict_captcha(image_path)
        set_pred(pred)
    return

def set_path(file_path: str):
    txtImage.delete(0, "end")
    txtImage.insert(0, file_path)

def set_image(file_path: str):
    img = ImageTk.PhotoImage(Image.open(file_path))
    img_width, img_height = img.width(), img.height()
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    x = (canvas_width - img_width) // 2
    y = (canvas_height - img_height) // 2
    canvas.create_image(x, y, anchor="nw", image=img)
    canvas.image = img

def set_pred(pred: str):
    txtPred.delete(0, "end")
    txtPred.insert(0, pred)

def image_predict():
    file_path = os.path.abspath(txtImage.get())
    pred = predict_captcha(model, file_path)
    set_pred(pred)

def drop_image(event):
    file_path = os.path.abspath(event.data)
    set_path(file_path)
    execute_pred(file_path)
    return

def gui():
    global captcha_name, train_data, model, window, selected_captcha, btnFileBrowser, txtImage, btnExec, canvas, txtPred

    window = TkinterDnD.Tk()  # 윈도우 창 생성
    window.drop_target_register(DND_FILES)
    window.dnd_bind('<<Drop>>', drop_image)

    # 윈도우 아이콘 설정
    icon_path = os.path.join(base_dir, "assets", "win.ico")
    window.iconbitmap(icon_path)  # 아이콘 설정
    window.title("Win Captcha Cracker")  # 제목 지정

    # 화면의 중앙에 위치시킴
    window_width = 600  # 창의 너비
    window_height = 200   # 창의 높이

    screen_width = window.winfo_screenwidth()  # 현재 화면 너비
    screen_height = window.winfo_screenheight()  # 현재 화면 높이

    x = (screen_width/2) - (window_width/2)  # x좌표
    y = (screen_height/2) - (window_height/2)  # y좌표
    window.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')  # 창의 크기와 위치를 설정
    window.resizable(False, False)  # 창 크기 비활성화
    window.grid_columnconfigure(1, weight=1)

    captcha_options = ["supreme_court", "gov24", "wetax"]
    selected_captcha = StringVar(window)
    selected_captcha.set(captcha_options[0])

    def update_captcha(*args):
        captcha_name = selected_captcha.get()
        captcha_id = captcha_name.upper()
        train_data = TrainInfo(id=captcha_id, name=captcha_name, images_base_dir=images_dir, model_path=model)
        model.train_data = train_data

    selected_captcha.trace_add("write", update_captcha)

    option_menu = OptionMenu(window, selected_captcha, *captcha_options)
    option_menu.config(width=11)
    option_menu.grid(row=0, column=0, padx=4, pady=4)

    txtImage = Entry(window)
    txtImage.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
    txtImage.bind("<Return>", lambda event: execute_pred(txtImage.get()))

    btnFileBrowser = Button(window, text='파일 찾기')  # 버튼 생성
    btnFileBrowser.grid(row=0, column=2, padx=4, pady=4)  # 버튼을 윈도우 창에 배치
    btnFileBrowser.config(command=select_image)

    btnExec = Button(window, width=11, height=4, text='실행')  # 버튼 생성
    btnExec.grid(row=1, column=0, padx=4, pady=4)  # 버튼을 윈도우 창에 배치
    btnExec.config(command=lambda: execute_pred(txtImage.get()))

    canvas = Canvas(window, width=300, height=80, border=2, relief='solid')  # 캔버스 생성
    canvas.config(bg='gray')
    canvas.grid(row=1, column=1, padx=4, columnspan=2)

    txtPred = Entry(window, width=8, font=('Arial 24'), justify='center')
    txtPred.grid(row=2, column=1, padx=4, columnspan=2)

    window.mainloop()  # 윈도우 창을 윈도우가 종료될 때까지 실행

if("__main__" == __name__):
    init()

    if len(argv) > 1:
        cli()

    else:
        from tkinter import Button, Entry, Canvas, filedialog
        from tkinterdnd2 import TkinterDnD, DND_FILES
        from PIL import Image, ImageTk
        from tkinter import StringVar, OptionMenu

        gui()

sys.exit(0)