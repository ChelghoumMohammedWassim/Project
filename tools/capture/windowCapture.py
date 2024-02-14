import win32gui,win32ui,win32con
import numpy as np


class WindowCapture:
    h=0
    w=0
    windowName=None
    hwnd=None
    
    def __init__(self,windowName=None) -> None:
        
        self.windowName=windowName

        
        if self.windowName == None:
            self.w= 1920
            self.h= 1080
        
        else:
            self.hwnd=win32gui.FindWindow(None,windowName)
            print('you are currently select '+ str(self.hwnd))
            window_rect=win32gui.GetWindowRect(self.hwnd)
            self.w= window_rect[2]-window_rect[0]
            self.h= window_rect[3]-window_rect[1]
        
        
    
    def get_screenshot(self):
        
        self.hwnd=0x140652
        
        #get the window image data
        wDC= win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0),(self.w,self.h),dcObj,(0,0),win32con.SRCCOPY)
        
        # save the ScreenShot
        dataBitMap.SaveBitmapFile(cDC,'debug.bmp')
        
        signedIntArray=dataBitMap.GetBitmapBits(True)
        image=np.fromstring(signedIntArray,dtype='uint8')
        image.shape=(self.h,self.w,4)
        
        #Free Ressource
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
        #drop alpha chanel
        image=image[...,:3]
        # image=np.ascontiguousarray(image)
        
        return image
    
    
    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, context):
            if win32gui.IsWindowVisible(hwnd):
                print('\t'+hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)
    
    