from django.urls import path
from recognition.views import ImageRecognitionView

urlpatterns = [
    # 现有的URL配置...
    path('api/recognition/analyze-with-nlp/', ImageRecognitionView.as_view()),
] 