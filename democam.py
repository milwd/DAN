import time 
import numpy as np
import cv2
import torch
from torchvision import transforms
from networks.dan import DAN


class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.ToPILImage(), 
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        checkpoint = torch.load('affecnet8_epoch5_acc0.6209.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0),cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)
        return faces

    def fer(self):

        fps_data = {'min':1000, 'max':0}
        cam = cv2.VideoCapture(0)

        while cam.isOpened() == True:
            ret, img0 = cam.read()
            assert ret == True

            faces = self.detect(img0)

            if len(faces) == 0:
                continue

            ##  single face detection
            x, y, w, h = faces[0]
            ## multi face detection 
            ## TODO

            img = img0[y:y+h, x:x+w]
            cv2.rectangle(img0, (x, y), (x+w, y+h), 255, 1) 

            img = self.data_transforms(img)
            img = img.view(1,3,224,224)
            img = img.to(self.device) 

            t1 = time.time()
            with torch.set_grad_enabled(False):
                out, _, _ = self.model(img)
                _, pred = torch.max(out,1)
                index = int(pred)
                label = self.labels[index]

                fps = round(1/(time.time()-t1), 3)
                if fps > fps_data['max']:
                    fps_data['max'] = fps
                elif fps < fps_data['min']:
                    fps_data['min'] = fps 

                print(f'emotion label: {label} \t\t in {fps}')
            
            cv2.imshow("hel", img0)
            if cv2.waitKey(1) == ord('q'):
                break 

        cam.release()
        cv2.destroyAllWindows()

        return fps_data
        

if __name__ == "__main__":

    model = Model()

    a = model.fer()

    print("OUTPUT:", a)    

