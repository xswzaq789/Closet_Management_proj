import io
from PIL import Image as im
import torch

from django.shortcuts import render
from django.views.generic.edit import CreateView

from .models import ImageModel
from .forms import ImageUploadForm

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

            path_hubconfig = "c:/Users/crid2/django_yolo_web/yolov5_code" # yolov5 폴더 루트
            path_weightfile = "c:/Users/crid2/django_yolo_web/yolov5_code/train_file/yolov5s.pt" # yolov5 가중치로 학습한 pt파일위치
            model = torch.hub.load(path_hubconfig, 'custom',
                                   path=path_weightfile, source='local'  )


            # 이미지 라벨 갯수 옵션 ( 보통 2개로 세팅 (상의,하의 ) , 사진이 1인 전신샷이라고 가정)
            model.max_det = 2

            # 라벨 지정 학률 (너무 낮은 확률이면 애매한 옷도 모두 지정해버림)
            model.conf = 0.6

            # 라벨링 된 옷 데이터만 따로 저장 기능



            results = model(img, size=640)


            # 크롭파일 이미지화 진행중
            crops = results.crop(save=True)  # cropped detections dictionary
            test01 = crops[0]
            test02 = crops[1]
            # 반환시 좌표로 넘파이 어레이로 반환 다시 이미지파일 변환 과정 필요





            # 추가 옷 종류만 json 파일로 표시 가능
            cloths_type = results.pandas().xyxy[0]['name'].to_json(orient='records')
            #test = results.pandas().xyxy[0] (라벨데이터 전체출력)

            # Results 업로드 이미지와 추론라벨 넘파이 결과값을 다시 이미지로 변환

            results.render()
            for img in results.imgs:
                img_base64 = im.fromarray(img)
                # 결과 저장 및 폴더지정
                img_base64.save("media/yolo_out/result.jpg", format="JPEG")
            inference_img = "/media/yolo_out/result.jpg"


            form = ImageUploadForm()
            context = {
                "form": form,
                "inference_img": inference_img,
                'cloths_type' : cloths_type,
                'test01' : test01 ,
                'test02' : test02



            }
            return render(request, 'image/imagemodel_form.html', context)

        else:
            form = ImageUploadForm()
        context = {
            "form": form
        }
        return render(request, 'image/imagemodel_form.html', context)




