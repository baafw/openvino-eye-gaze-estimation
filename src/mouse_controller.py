'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui

class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'faster':0,'fast':1, 'slow':10, 'medium':5}

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.prevx = 0
        self.prevy = 0

    def move(self, x, y):
        
        # if(abs( x - self.prevx) < 0.09 and abs( y - self.prevy) < 0.09):
        #     print(abs( x - self.prevx))
        #     self.prevx = x
        #     self.prevy = y
            
        #     return
        
        mx, my = pyautogui.position()
        xdistance = x*self.precision #if  x*self.precision > 5 else 0
        ydistance = -1*y*self.precision #if -1*y*self.precision > 5 else 0
        
        if(pyautogui.onScreen(mx + xdistance, my + ydistance)):
            pyautogui.moveRel(xdistance,ydistance, duration=self.speed)

    def movexy(self, x, y):
        pyautogui.move(x, y, duration=self.speed)
