from ultralytics import YOLO
import cv2

model = YOLO(
    'weights/wasteland_model/WtE_Predictor/v3_final_refinement/weights/best.pt'

)

cv2.namedWindow('AI Prediction', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('AI Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for x in range(1, 15):
    if(x!=4):
        results = model.predict(
            source=f'garbage_testing_dataset/f{x}.jpg',
            conf=0.0001
        )

        for r in results:
            im_array = r.plot(line_width=1, font_size=0.2, labels=True,masks=False)

            cv2.imshow('AI Prediction', im_array)

            cv2.waitKey(0)   

cv2.destroyAllWindows()
