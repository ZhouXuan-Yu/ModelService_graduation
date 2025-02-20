from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    path('knowledge/', views.knowledge_page, name='knowledge_page'),
    path('add-knowledge/', views.add_knowledge, name='add_knowledge'),
    path('get-knowledge/', views.get_knowledge, name='get_knowledge'),
    path('delete-knowledge/', views.delete_knowledge, name='delete_knowledge'),
    path('upload-file/', views.upload_file, name='upload_file'),
    path('api/route/plan/', views.route_plan, name='route_plan'),
]