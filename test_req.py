import requests

try:
    with open('test.jpg', 'rb') as f:
        files = {'image': ('test.jpg', f, 'image/jpeg')}
        res = requests.post('http://localhost:8000/api/caption', files=files)
        print(res.json())
except Exception as e:
    print("Error:", e)
