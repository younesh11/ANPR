import pytesseract 

def imgToText(img):

    text = pytesseract.image_to_string(img, config='--psm11')
    print (text)

    return text



