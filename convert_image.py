import base64

def encode_image():
    with open("/home/bruno-rg/Documents/object-detection-api/images/AX_7170.jpeg", "rb") as image_file:
        data = base64.b64encode(image_file.read())
    image_file.close()

    with open("/home/bruno-rg/Documents/object-detection-api/images/AX_7170.txt", "w", encoding='utf-8') as txt_file:
        txt_file.write(str(data))
    txt_file.close()
    print(data)

def decode_image():
    with open("/home/bruno-rg/Documents/object-detection-api/images/AX_7170.txt", "r") as txt_file:
        data = txt_file.read()
        

        txt_file.close()

        print(data)

decode_image()
#encode_image()