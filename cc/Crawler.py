from playwright.sync_api import sync_playwright, Frame
from PIL import Image

# Assuming TrainModel is defined in a module named train_model
from cc.Core import *
import Core
import time, os

class Engine:

    def __init__(self, train_data: TrainData):
        self.train_data = train_data
        self.browser = None
        self.page = None        
        self.url = "https://www.scourt.go.kr/portal/information/events/search/search.jsp"
        self.btn_selector = "div.breakDiv > a.blueBtn"
        self.img_selector = "#captcha"
        self.count = 1
        self.exec = lambda **args: any
        self.openPage()

    def __exit__(self, exc_type, exc_value, traceback):
        self.closePage()

    def openPage(self):
        self.browser = p.chromium.launch()
        self.page = browser.new_page()
        self.page.goto(self.url)

    def closePage(self):
        if self.page:
            self.page.close()
            self.page = None

        if self.browser:
            self.browser.close()
            self.browser = None

    def loadCaptcha(self, count: int = 1):
        return Exception("Not implemented")

    def saveCaptcha(self, image_path: str):
        fileName = os.path.basename(image_path)

        with Image.open(image_path) as img:
            savePath = os.path.join("images", "supreme_court", "scrap", fileName)
            width, height = img.size
            cropped_img = img.crop((2, 2, width - 1, height - 1))
            cropped_img.save(savePath)

    def run(self):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(self.url)
            print(page.title())
            self.saveCaptcha(frame=page.frame_locator('iframe'), count=10)
            browser.close()

url = "https://www.scourt.go.kr/portal/information/events/search/search.jsp"
btn_selector = "div.breakDiv > a.blueBtn"
img_selector = "#captcha"
i = 0

def saveCaptcha(frame: Frame, count: int = 1):
    for i in range(count):
        frame.locator(btn_selector, has_text="새로고침").click()
        time.sleep(1)
        
        fileName = f"{int(time.time())}.png"
        tempPath = os.path.join("temp", fileName)
        frame.locator(img_selector).screenshot(path=tempPath)

        with Image.open(tempPath) as img:
            savePath = os.path.join("images", "supreme_court", "scrap", fileName)
            width, height = img.size
            cropped_img = img.crop((2, 2, width - 1, height - 1))
            cropped_img.save(savePath)
            print(f"Saved {i}: {savePath}")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url)
    print(page.title())
    saveCaptcha(frame=page.frame_locator('iframe'), count=10)



        # cropped_img = img.crop((2, 2, img. 1, 1))
        # cropped_img.save("sc_cropped.png")

    # frames = page.frames
    # for frame in frames:
    #     print(f"Frame name: {frame.name}, Frame URL: {frame.url}")
    #     if "mysafind.jsp" in frame.url:

            # element = frame.locator(selector)
            # if element:
            #     element.screenshot(path="sc.png")
            #     print(element.inner_text())
            # framePage = frame.page
            # element = framePage.locator(selector)
            # if element:
            #     element.screenshot(path="sc.png")
            #     print(element.inner_text())
            # element = frame.page.query_selector(selector)
            # if element:
            #     element.screenshot(path="sc.png")
            #     print(element.inner_text())

    # element = page.query_selector("selector")
    # print(element.inner_text())
    browser.close()