from django.urls import re_path
from tracking.views import VideoTrackingConsumer

websocket_urlpatterns = [
    re_path(r'ws/tracking/(?P<tracking_id>\w+)/$', VideoTrackingConsumer.as_asgi()),
] 