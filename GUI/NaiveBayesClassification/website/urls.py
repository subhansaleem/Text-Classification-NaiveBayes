
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('tfidf',views.tfidf, name='tfidf'),
    path('noun', views.noun, name='noun'),
    path('lexical', views.lexical, name='lexical'),
    path('mix', views.mix, name='mix'),
]
