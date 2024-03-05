import os
from contextlib import nullcontext
from rest_framework import serializers
from django.urls import reverse
from user.serializers import UserSerializer
from audiobook.models import Book
from .models import Post, User, Comment, Inquiry
from config.settings import AWS_S3_CUSTOM_DOMAIN, MEDIA_URL, FILE_SAVE_POINT, MEDIA_ROOT


class PostOldSerializer(serializers.ModelSerializer):
    user_nickname = serializers.CharField(
        source='user.nickname', read_only=True)
    post_created_date = serializers.DateTimeField(
        format="%Y-%m-%dT%H:%M:%S.%fZ", read_only=True)
    post_updated_date = serializers.DateTimeField(
        format="%Y-%m-%dT%H:%M:%S.%fZ", read_only=True, allow_null=True)

    class Meta:
        model = Post
        fields = ['post_id', 'post_title', 'post_content', 'user_id',
                  'user_nickname', 'post_created_date', 'post_updated_date']

    def save(self, **kwargs):
        book_id = self.context.get('book_id')
        user_id = self.context.get('user_id')
        # 현재 로그인 유저
        user = User.objects.get(pk=user_id)
        # 현재 선택된 책
        book = Book.objects.get(pk=book_id)
        self.validated_data['user'] = user
        self.validated_data['book'] = book
        return super().save(**kwargs)

    def to_representation(self, instance):
        response = super().to_representation(instance)
        response['user'] = UserSerializer(instance.user).data
        # response['book'] = BookSerializer(instance.book).data
        return response

class CommentSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = Comment
        fields = ['comment_id', 'comment_content', 'user']

    def save(self, **kwargs):
        post_id = self.context['post_id']
        user_id = self.context['user_id']
        # 현재 로그인 유저
        user = User.objects.get(pk=user_id)
        # 현재 선택된 게시물
        post = Post.objects.get(pk=post_id)
        self.validated_data['user'] = user
        self.validated_data['post_id'] = post_id  # Store post_id directly, not the object

        return super().save(**kwargs)

    def to_representation(self, instance):
        response = super().to_representation(instance)
        return response



class PostSerializer(serializers.ModelSerializer):
    user_nickname = serializers.CharField(source='user.nickname', read_only=True)
    post_created_date = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%S.%fZ", read_only=True)
    post_updated_date = serializers.DateTimeField(format="%Y-%m-%dT%H:%M:%S.%fZ", read_only=True, allow_null=True)
    comments = CommentSerializer(many=True, read_only=True)  # Assuming you have a CommentSerializer

    class Meta:
        model = Post
        fields = ['post_id', 'post_title', 'post_content', 'user_id', 'user_nickname', 'post_created_date', 'post_updated_date', 'comments']


    def save(self, **kwargs):
        book_id = self.context.get('book_id')
        user_id = self.context.get('user_id')
        # 현재 로그인 유저
        user = User.objects.get(pk=user_id)
        # 현재 선택된 책
        book = Book.objects.get(pk=book_id)
        self.validated_data['user'] = user
        self.validated_data['book'] = book
        return super().save(**kwargs)

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret['user'] = UserSerializer(instance.user).data
        return ret

class CommentOldSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ['comment_id', 'comment_content']

    def save(self, **kwargs):
        post_id = self.context['post_id']
        user_id = self.context['user_id']
        # 현재 로그인 유저
        user = User.objects.get(pk=user_id)
        # 현재 선택된 게시물
        post = Post.objects.get(pk=post_id)
        self.validated_data['user'] = user
        self.validated_data['post'] = post

        return super().save(**kwargs)

    def to_representation(self, instance):
        response = super().to_representation(instance)
        response['user'] = UserSerializer(instance.user).data
        response['post'] = PostSerializer(instance.post).data
        return response


class BookSerializer(serializers.ModelSerializer):
    post_set = PostOldSerializer(many=True, read_only=True)

    class Meta:
        model = Book
        fields = '__all__'
        
    def create(self, validated_data):
        if not validated_data.get('book_view_count'):
            validated_data['book_view_count'] = {"1": 0, "2": 0, "3": 0,"4": 0,"5": 0,"6": 0,
                                                 "7": 0, "8": 0, "9": 0,"10": 0,"11": 0,"12": 0}
        return super().create(validated_data)

    def update(self, instance, validated_data):
        if validated_data.get('book_likes'):
            instance.book_likes += validated_data.get('book_likes')
        if validated_data.get('book_view_count'):
            month = self.context['month']
            instance.book_view_count[str(month)] += 1
        instance.save()
        return instance

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret['post_set'] = PostSerializer(
            instance.post_set.all(), many=True).data
        return ret



class InquirySerializer(serializers.ModelSerializer):
    detail_url = serializers.SerializerMethodField()  # get_detail_url() 의 return
    inquiry_created_date = serializers.DateTimeField(
        format='%Y-%m-%d %H:%M', read_only=True)
    inquiry_answered_date = serializers.DateTimeField(
        format='%Y-%m-%d %H:%M', read_only=True)

    class Meta:
        model = Inquiry
        fields = '__all__'

    def get_detail_url(self, obj):
        return reverse('manager:inquiry_detail', kwargs={'inquiry_id': obj.inquiry_id})
