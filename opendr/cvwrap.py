try:
    import cv2
except:
    raise Exception('Failed to import cv2 from the OpenCV distribution. Please install OpenCV with Python support. OpenCV may either be installed from http://opencv.org or installed with package managers such as Homebrew (on Mac, http://brew.sh) or apt-get (on Ubuntu or Debian).')