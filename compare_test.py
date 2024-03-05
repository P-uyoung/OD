import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

import timeit
from django.test import RequestFactory
from django.contrib.auth import get_user_model  # Import the get_user_model function
from community.views import BookShareContentPostDetailHtmlOldVersion, BookShareContentPostDetailHtml

# Use get_user_model to retrieve the custom user model
User = get_user_model()

# Create or retrieve a test user
# Make sure to replace 'test_username' and 'test_password' with actual values
test_user, _ = User.objects.get_or_create(username='test_username', defaults={'password': 'test_password'})

request_factory = RequestFactory()

pk_test_value = 1

def view_test_old():
    request = request_factory.get(f'/books/share/content/post/detail/{pk_test_value}/')
    # Manually authenticate the user
    request.user = test_user
    response = BookShareContentPostDetailHtmlOldVersion.as_view()(request, pk=pk_test_value)

def view_test_new():
    request = request_factory.get(f'/books/share/content/post/detail/{pk_test_value}/')
    # Manually authenticate the user
    request.user = test_user
    response = BookShareContentPostDetailHtml.as_view()(request, pk=pk_test_value)

iter = 10

# timeit으로 성능 측정
old_time = timeit.timeit('view_test_old()', globals=globals(), number=iter)
new_time = timeit.timeit('view_test_new()', globals=globals(), number=iter)

print(f"Old Version Time: {round( old_time/iter*1000, 2)} ms")
print(f"New Version Time: {round( new_time/iter*1000, 2)} ms")
