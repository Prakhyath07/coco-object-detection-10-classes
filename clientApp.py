from segmentation import get_yolov5, get_image_from_bytes
from flask import Flask, render_template, request, Response, jsonify
import os
import io
from flask_cors import CORS, cross_origin
from com_ineuron_utils.utils import decodeImage
from PIL import Image
import base64
from com_ineuron_utils.utils import encodeImageIntoBase64

app = Flask(__name__)

model = get_yolov5()

RENDER_FACTOR = 35

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

CORS(app)


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "file.jpg"
        # modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        # self.objectDetection = Detector(self.filename)


# def run_inference(img_path='file.jpg'):
#     # run inference using detectron2
#     result_img = detector.inference(img_path)

#     # clean up
#     try:
#         os.remove(img_path)
#     except:
#         pass

#     return result_img



@app.route("/")
def home():
    # return "Landing Page"
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        input_image = get_image_from_bytes("file.jpg")
        results = model(input_image)
        # print(results)
        results.render()  # updates results.imgs with boxes and labels
        # print(results.render())
        for img in results.ims[:1]:
            bytes_io = io.BytesIO()
            
            img_base64 = Image.fromarray(img)
            # print(type(img))
            
            # img_base64 = base64.b64encode(img)
            # print(type(img_base64))
            img_base64.save(bytes_io, format="jpeg")
            img_base64.save("color_img.jpg")
            opencodedbase64 = encodeImageIntoBase64("color_img.jpg")

            result = {"image" : opencodedbase64.decode('utf-8') }
            
            
            # result = {"image" : img_base64.decode('utf-8') }
            
        # return Response(content=bytes_io.getvalue(),media_type="image/jpeg")
        # return jsonify(result)

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"
    return jsonify(result)


# port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    port = 9000
    app.run(host='127.0.0.1', port=port)
