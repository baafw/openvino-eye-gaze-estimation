'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import time
from inference import Network

class GazeEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_xml = model_name
        self.device =  device
        self.extensions = extensions
         # Initialise the class
        self.infer_network = Network()
        #raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.infer_network.load_model(self.model_xml, self.device, self.extensions)
        #raise NotImplementedError

    def predict(self, left_eye_image, right_eye_image, headpose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.infer_network.exec_net(headpose_angles, left_eye_image,right_eye_image)

        # Wait for the result
        if self.infer_network.wait() == 0:
            # end time of inference
            end_time = time.time()
            result = (self.infer_network.get_output())[self.infer_network.output_blob]
            return result


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, frame, face, left_eye_point, right_eye_point, print_flag=True):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

       Blob in the format [BxCxHxW] where:
        B - batch size
        C - number of channels
        H - image height
        W - image width
        with the name left_eye_image and the shape [1x3x60x60].
        Blob in the format [BxCxHxW] where:
        B - batch size
        C - number of channels
        H - image height
        W - image width
        with the name right_eye_image and the shape [1x3x60x60].
        Blob in the format [BxC] where:
        B - batch size
        C - number of channels
        with the name head_pose_angles and the shape [1x3].

        '''
        
        lefteye_input_shape =  [1,3,60,60] #self.infer_network.get_input_shape()
        righteye_input_shape = [1,3,60,60] #self.infer_network.get_next_input_shape(2)

        # crop left eye
        x_center = left_eye_point[0]
        y_center = left_eye_point[1]
        width = lefteye_input_shape[3]
        height = lefteye_input_shape[2]
        # ymin:ymax, xmin:xmax 
        facewidthedge = face.shape[1]
        faceheightedge = face.shape[0]
        
        # check for edges to not crop
        ymin = int(y_center - height//2) if  int(y_center - height//2) >=0 else 0 
        ymax = int(y_center + height//2) if  int(y_center + height//2) <=faceheightedge else faceheightedge

        xmin = int(x_center - width//2) if  int(x_center - width//2) >=0 else 0 
        xmax = int(x_center + width//2) if  int(x_center + width//2) <=facewidthedge else facewidthedge


        left_eye_image = face[ymin: ymax, xmin:xmax]
        # print out left eye to frame
        if(print_flag):
            frame[150:150+left_eye_image.shape[0],20:20+left_eye_image.shape[1]] = left_eye_image
        # left eye [1x3x60x60]
        p_frame_left = cv2.resize(left_eye_image, (lefteye_input_shape[3], lefteye_input_shape[2]))
        p_frame_left = p_frame_left.transpose((2,0,1))
        p_frame_left = p_frame_left.reshape(1, *p_frame_left.shape)

        # crop right eye
        x_center = right_eye_point[0]
        y_center = right_eye_point[1]
        width = righteye_input_shape[3]
        height = righteye_input_shape[2]
        # ymin:ymax, xmin:xmax 
        # check for edges to not crop
        ymin = int(y_center - height//2) if  int(y_center - height//2) >=0 else 0 
        ymax = int(y_center + height//2) if  int(y_center + height//2) <=faceheightedge else faceheightedge

        xmin = int(x_center - width//2) if  int(x_center - width//2) >=0 else 0 
        xmax = int(x_center + width//2) if  int(x_center + width//2) <=facewidthedge else facewidthedge

        right_eye_image =  face[ymin: ymax, xmin:xmax]
        # print out left eye to frame
        if(print_flag):
            frame[150:150+right_eye_image.shape[0],100:100+right_eye_image.shape[1]] = right_eye_image
            
        # right eye [1x3x60x60]
        p_frame_right = cv2.resize(right_eye_image, (righteye_input_shape[3], righteye_input_shape[2]))
        p_frame_right = p_frame_right.transpose((2,0,1))
        p_frame_right = p_frame_right.reshape(1, *p_frame_right.shape)


        #headpose_angles

        return frame, p_frame_left, p_frame_right
        #raise NotImplementedError

    def preprocess_output(self, outputs, image,facebox, left_eye_point, right_eye_point,print_flag=True, threshold = 0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.
        Output layer name in Inference Engine format:
        gaze_vector
        '''
        x = outputs[0][0]
        y = outputs[0][1]
        z = outputs[0][2]
        #Draw output
        if(print_flag):
            cv2.putText(image,"x:"+str('{:.1f}'.format(x*100))+",y:"+str('{:.1f}'.format(y*100))+",z:"+str('{:.1f}'.format(z)) , (20, 100), 0,0.6, (0,0,255), 1)

            #left eye
            xmin,ymin,_,_ = facebox
            x_center = left_eye_point[0]
            y_center = left_eye_point[1]
            left_eye_center_x = int(xmin + x_center)
            left_eye_center_y = int(ymin + y_center)
            #right eye
            x_center = right_eye_point[0]
            y_center = right_eye_point[1]
            right_eye_center_x = int(xmin + x_center)
            right_eye_center_y = int(ymin + y_center)

            cv2.arrowedLine(image, (left_eye_center_x,left_eye_center_y), (left_eye_center_x + int(x*100),left_eye_center_y + int(-y*100)), (255, 100, 100), 5)
            cv2.arrowedLine(image, (right_eye_center_x,right_eye_center_y), (right_eye_center_x + int(x*100),right_eye_center_y + int(-y*100)), (255,100, 100), 5)

        return image, [x, y, z]
