import gdown

url = 'https://drive.google.com/u/0/uc?id=1b4vyiNIghGV9nwMnMki5mpC6kujLHP11&export=download'
output = 'models.zip'
gdown.download(url, output, quiet=False)


# cd models
# cp *.pth ../
# cp *.py ../
# cd ..
# rm -rf models
# rm -rf models.zip