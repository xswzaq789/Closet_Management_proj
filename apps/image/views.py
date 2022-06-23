import io
from PIL import Image as im
import torch

from django.shortcuts import render
from django.views.generic.edit import CreateView

from .models import ImageModel
from .forms import ImageUploadForm

import urllib.request
import json

#  0) 절대경로 path 설정
abs_path = 'C:/Users/crid2/django_yolo_web/'

# 1) 컬러함수 필요 라이브러리
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# 2) 처음 서버 작동시 로드됨
model = load_model(abs_path + 'yolov5_code/train_file/color.h5')

# 3) 연결코드
def color_classfication(numpy_value) :
        global color_result
        crop_image = im.fromarray(numpy_value , mode=None)
        crop_image.save('media/crop/crop0.jpg')
        img_src = 'media/crop/crop0.jpg'
        test_img = image.load_img(img_src, target_size=(200, 200))
        x = image.img_to_array(test_img)
        x = np.expand_dims(x, axis=0)
        image_ = np.vstack([x])
        classes = model.predict(image_, batch_size=10)
        print('##### cloths image result ####')
        print()
        print('pred - ', classes[0])
        print('color :' , np.argmax(classes[0]))
        print()
        color_result = int(np.argmax(classes[0]))


class UploadImage(CreateView):
    model = ImageModel
    template_name = 'image/imagemodel_form.html'
    fields = ["image"]

    def post(self, request, *args, **kwargs):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid(): # is_valid() 메서드 데이터의 유효성 검사하는 역활
            img = request.FILES.get('image')
            img_instance = ImageModel(
                image=img
            )
            img_instance.save() # 넘파이나 바이너리로 저장하는 기능
            uploaded_img_qs = ImageModel.objects.filter().last()
            img_bytes = uploaded_img_qs.image.read()
            img = im.open(io.BytesIO(img_bytes))


            path_hubconfig = abs_path + "yolov5_code" # yolov5 폴더 루트
            path_weightfile = abs_path + "yolov5_code/train_file/yolov5s.pt" # yolov5 가중치로 학습한 pt파일위치
            model = torch.hub.load(path_hubconfig, 'custom', path=path_weightfile, source='local')


            # 이미지 라벨 갯수 옵션 ( 보통 2개로 세팅 (상의,하의 ) , 사진이 1인 전신샷이라고 가정)
            model.max_det = 1
            # 라벨 지정 학률 (너무 낮은 확률이면 애매한 옷도 모두 지정해버림)
            model.conf = 0.25


            results = model(img, size=640)

            # 라운딩 박스 세팅수와 crops 이미지 갯수가 불일치 에러
            crops = results.crop(save=False)  # cropped detections dictionary , True 이미지 생성
            # model.max_det = 1 개일때 객체가 0이면 'None'값을 반환
            try :
                test01 = crops[0]['label']

                # 4) 크롭된 이미지 색깔판별 함수 호출 color_classfiaction()
                # [:,:,::-1] BGR -> RGB 값으로 전환 넘파이를 이미지 저장시 색상반전을 보정역활
                color_classfication(crops[0]['im'][:, :, ::-1])

                print('black: 0, blue: 1, green: 2, pattern: 3, red: 4, white: 5')
                print(test01)
                print(crops[0]['im'].shape)

            except IndexError :
                print('NO detect , try again ')
                test01 = 'No detect'





            # 추가 옷 종류만 json 파일로 표시 가능
            cloths_type = results.pandas().xyxy[0]['name'].to_json(orient='records')
            print(cloths_type)
            #test = results.pandas().xyxy[0] (라벨데이터 전체출력)

            # Results 업로드 이미지와 추론라벨 넘파이 결과값을 다시 이미지로 변환

            results.render()
            for img in results.imgs:
                img_base64 = im.fromarray(img)
                # 결과 저장 및 폴더지정
                img_base64.save("media/yolo_out/result.png" , format="JPEG")
            inference_img = "/media/yolo_out/result.png"

            # 딕셔너리를 json으로 변환
            import json
            cloths_data = {'cloths_tpye' : cloths_type ,
                            'color_code' : color_result}

            cloths_json = json.dumps(cloths_data)





            form = ImageUploadForm()
            context = {
                "form": form,
                "inference_img": inference_img,
                'cloths_type' : cloths_type,
                'cloths_json' : cloths_json

            }
            return render(request, 'image/imagemodel_form.html', context)

        else:
            form = ImageUploadForm()
        context = {
            "form": form
        }
        return render(request, 'image/imagemodel_form.html', context)




