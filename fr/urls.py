from django.urls import path
from . import views

urlpatterns = [
    path("", views.index),
    path("enroll/", views.enroll),
    path("train/", views.train),
    path("recognize/", views.recognize),
    path("download_attendance/", views.download_attendance),
    path("offices/", views.list_offices),
    path("offices/delete/", views.delete_office),
    path("offices/rename/", views.rename_office),
    path("profile/update/", views.update_profile),
    path("profile/<str:office>/<str:employee>/", views.profile_image),

    # NEW ADMIN SYSTEM
    path("admin-panel/", views.admin_offices, name="admin_offices"),
    path("admin-panel/<str:office>/", views.admin_employees, name="admin_employees"),
    path("admin-panel/<str:office>/<str:employee>/", views.admin_employee_logs, name="admin_employee_logs"),
]
