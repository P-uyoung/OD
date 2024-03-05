from django.db import models
from audiobook.models import Book

from user.models import User


class Post(models.Model):
    post_id = models.AutoField(primary_key=True)
    post_title = models.CharField(max_length=255)
    post_content = models.TextField()
    post_created_date = models.DateTimeField(auto_now_add=True)
    post_updated_date = models.DateTimeField(null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)


class Comment(models.Model):
    comment_id = models.AutoField(primary_key=True)
    comment_content = models.TextField()
    comment_created_date = models.DateTimeField(auto_now_add=True)
    comment_updated_date = models.DateTimeField(null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    post = models.ForeignKey(Post, related_name='comments', on_delete=models.CASCADE)


class BookRequest(models.Model):
    request_id = models.AutoField(primary_key=True)
    request_isbn = models.CharField(max_length=255)
    request_count = models.IntegerField(default=0)


class UserRequestBook(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    request = models.ForeignKey(BookRequest, on_delete=models.CASCADE)


class Inquiry(models.Model):
    inquiry_id = models.AutoField(primary_key=True)
    inquiry_category = models.CharField(max_length=255, default="Others")
    inquiry_title = models.CharField(max_length=255)
    inquiry_content = models.TextField(null=True, blank=True)
    inquiry_response = models.TextField(null=True, blank=True)
    inquiry_created_date = models.DateTimeField(auto_now_add=True)
    inquiry_answered_date = models.DateTimeField(null=True, blank=True)
    inquiry_is_answered = models.BooleanField(default=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
