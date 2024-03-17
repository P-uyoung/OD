from django.urls import path, include
from . import views

app_name = 'community'

urlpatterns = [
    # APIs
    path('api/user/', views.UserList.as_view(), name='user_list'),
    path('api/user/<int:pk>', views.UserDetail.as_view(), name='user_detail'),
    path('api/book/', views.BookList.as_view(), name='book_list'),
    path('api/book/<int:pk>', views.BookDetail.as_view(), name='book_detail'),
    path('api/post/', views.PostList.as_view(), name='post_list'),
    path('api/post/<int:pk>', views.PostDetail.as_view(), name='post_detail'),
    path('api/comment/', views.CommentList.as_view(), name='comment_list'),
    path('api/comment/<int:pk>',
         views.CommentDetail.as_view(), name='comment_detail'),
    path('api/inquiry/', views.InquiryList.as_view(), name='inquiry_list'),
    path('api/inquiry/<int:pk>',
         views.InquiryDetail.as_view(), name='inquiry_detail'),
    path('api/faq/', views.FAQList.as_view(), name='faq_list'),
    path('api/faq/<int:pk>', views.FAQDetail.as_view(), name='faq_detail'),

    # html render
    # 토론
    path('books/share/', views.BookShareHtml.as_view(), name='book_share'),
    path('books/share/content/<int:pk>',
         views.BookShareContentHtml.as_view(), name='book_share_content'),
    path('books/share/content/post/', views.BookShareContentPostHtml.as_view(),
         name='book_share_content_post'),
    path('books/share/content/post/detail/<int:pk>',
         views.BookShareContentPostDetailHtml.as_view(), name='book_share_content_post_detail'),


    # 신규 도서 신청
    # http://127.0.0.1:8000/community/books/search
    path('books/search/', views.BookSearchView.as_view(), name='book_search'),
    path('api/books/search/<str:isbn>/',
         views.BookRequestAPI.as_view(), name='book_complete'),

    # 1:1 문의
    path('books/inquiry/', views.InquiryPostHtml.as_view(), name='book_inquiry'),
    path('books/inquiry/complete/', views.InquiryPostCompleteHtml.as_view(),
         name='book_inquiry_complete'),

    # FAQ
    path('books/faq/', views.FAQHtml.as_view(), name='book_faq'),

    # 좋아요
    path('books/like/', views.BookLikeView.as_view(), name='book_like')
]
