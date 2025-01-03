from playwright.sync_api import sync_playwright, Frame
from PIL import Image

# Assuming TrainModel is defined in a module named train_model
from core import train_model as TrainModel
import time, os

class Engine:
    def __init__(self, train_model: TrainModel):
        self.url = "https://www.scourt.go.kr/portal/information/events/search/search.jsp"
        self.btn_selector = "div.breakDiv > a.blueBtn"
        self.img_selector = "#captcha"
        self.i = 0

    def saveCaptcha(self, frame: Frame, count: int = 1):
        for i in range(count):
            frame.locator(self.btn_selector, has_text="새로고침").click()
            time.sleep(1)
            
            fileName = f"{int(time.time())}.png"
            tempPath = os.path.join("temp", fileName)
            frame.locator(self.img_selector).screenshot(path=tempPath)

            with Image.open(tempPath) as img:
                savePath = os.path.join("images", "supreme_court", "scrap", fileName)
                width, height = img.size
                cropped_img = img.crop((2, 2, width - 1, height - 1))
                cropped_img.save(savePath)
                print(f"Saved {i}: {savePath}")

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