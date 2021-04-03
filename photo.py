import cv2


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    name = input("nhap Ten thu muc:\n")
    # Path
    cam = cv2.VideoCapture(0)
    sampleNum = 0
    takephoto = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x=200
        y=110
        w=230
        h=230
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if takephoto == 1:
            sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
            cv2.imwrite("E:/Face-Mask-Detection-master/dataset/" + name + "/" + str(sampleNum) + ".jpg",
                            img[y:y + h, x:x + w])
            # display the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(sampleNum), (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Nhan nut C de chup anh', (200, 50), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', img)
        # wait for 100 miliseconds
        if cv2.waitKey(10) & 0xFF == ord('c'):
            if takephoto==0:
                takephoto = 1
            else:
                takephoto = 0
        # break if the sample number is morethan 100
        if sampleNum > 999:
            break
    cam.release()
    cv2.destroyAllWindows()


TakeImages()