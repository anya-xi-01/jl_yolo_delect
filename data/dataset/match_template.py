import cv2


def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def match_template(im0, im1):
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    im0_A, im0_B = im0.shape
    im1_A, im1_B = im1.shape

    h = max(0, im1_A-im0_A)
    w = max(0, im1_B-im0_B)

    if h > 0 or w > 0:
        h = int((h/2)+1)
        w = int((w/2)+1)
        im0 = cv2.copyMakeBorder(im0, h, h, w, w, cv2.BORDER_CONSTANT, 0)

    ret = cv2.matchTemplate(im0, im1, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ret)
    return 0 if min_val is None else 1 - min_val


def match_and_compare(im0, im1, im2):
    im1_similar = match_template(im0, im1)
    im2_similar = match_template(im0, im2)
    return im1_similar if im1_similar>im2_similar else im2_similar


if __name__ == '__main__':
    im0 = cv2.imread("/home/anya-xi/machinelearning/code/jl_yolo_delect/data/dataset/temp/crop/666_0_crop4.jpg")
    im1 = cv2.imread("/home/anya-xi/图片/0601_2_1.png")

    ret = match_template(im0, im1)



    print(ret) # 0.5050142705440521  0.0
