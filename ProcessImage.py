import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import csv
from pathlib import Path
import time
import logging
import progressbar


class ProcessImage:
    '''The processImage class to process images and extract useful information

    Args:
        imagePath (string): image path or image list directory to load images from it
        top (int):  total pixels should croped from top side of image
        right (int):  total pixels should croped from right side of image
        bottom (int):  total pixels should croped from bottom side of image
        left (int):  total pixels should croped from left side of image
    '''

    _CIRCLE_MASK_COLOR = [0, 0, 255]
    _OIL_MASK_COLOR = [255, 0, 0]
    _PLATE_MASK_COLOR = [0, 255, 0]

    def __init__(self, imagePath, top=0, right=0, bottom=0, left=0, plateLow=200, oilHigh=70, showFilePreview=False):
        '''The processImage class to process images and extract useful information

        Args:
            imagePath (string): image path or image list directory to load images from it
            top (int):  total pixels should croped from top side of image
            right (int):  total pixels should croped from right side of image
            bottom (int):  total pixels should croped from bottom side of image
            left (int):  total pixels should croped from left side of image
        '''
        self.imagePath = imagePath
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
        self.plateLow = plateLow
        self.oilHigh = oilHigh

        self.showFilePreview = showFilePreview

    def getFilesInPath(self):
        '''
        Get list of files recursivly in the imagePath and retun sorted list of file path list as array
        '''
        files = []
        if os.path.isdir(self.imagePath):
            os.chdir(self.imagePath)
            cwd = os.getcwd()
            files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(
                cwd) for f in filenames if os.path.splitext(f)[1] == '.jpg']

        elif os.path.isfile(self.imagePath):
            files = [self.imagePath]
        else:
            logging.error("It is a special file (socket, FIFO, device file)")
        files.sort()
        return files

    def getImageAtPath(self, filePath):
        '''
        Read image from file path and crop not needed pixels from it and return it and also calculate total pixels in image and set it as self.totalPixels
        '''
        img = cv2.imread(filePath)

        img = img[self.top: -self.bottom if self.bottom >
                  0 else img.shape[0], self.left: -self.right if self.right >
                  0 else img.shape[1]]

        self.totalPixels = img.shape[0] * img.shape[1]

        return img

    def maskCirclsImage(self, img):
        '''
        Try to find circles in image and add a circle object on it in the image and return it
        also calculate total circles on image and total pixels used by circles
        '''

        # Generate gray for image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Run gaussianBlur filter on the image for better edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Create threshold for canny edge detector
        threshold = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 3)
        # create a copy of image to add circles on it
        newImg = img.copy()

        # Canny Edges
        edges = cv2.Canny(threshold, 10, 250)

        # Find circles from founded edges by houghCircles detector
        # To find more circles you should change params in this function
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
                                   1, 30, param1=50, param2=30, minRadius=0, maxRadius=80)

        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(newImg, (i[0], i[1]), i[2],
                       self._CIRCLE_MASK_COLOR, thickness=-1)

        self.totalCirclePixels = np.sum(newImg[:, :, 2] == 255)
        self.totalCircles = len(circles[0, :])

        return newImg

    def maskedOilImage(self, img):
        '''
        Find oil colors pixels and convert its color to different color to find diffrence easier and also calculate total pixels for it
        also set totalOilPixels
        '''
        # convert RGB color mode to HSV to calculate values without hue and saturation of image
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        valueChannel = hsv[:, :, 2]
        bw = 255*np.uint8(valueChannel < (self.oilHigh))
        imgNpArray = np.copy(img)
        indices = np.where(bw == 255)
        imgNpArray[indices[0], indices[1], :] = self._OIL_MASK_COLOR

        self.totalOilPixels = np.sum(bw == 255)

        return imgNpArray

    def maskedPlateImage(self, img):
        '''
        Find plate colors pixels and convert its color to different color to find diffrence easier and also calculate total pixels for it
        NOTE: This function should run after maksCircles because our circles have the same color with plate color
        '''
        # Get image green color channel
        greenChannel = img[:, :, 1]
        bw = 255*np.uint8(greenChannel > (self.plateLow))
        imgNpArray = np.copy(img)
        indices = np.where(bw == 255)
        imgNpArray[indices[0], indices[1], :] = self._PLATE_MASK_COLOR
        self.totalPlatePixels = np.sum(bw == 255)
        return imgNpArray

    def getTotalCirclePixels(self):
        '''
        Get Total circles pixels for current in processing image
        '''
        return self.totalCirclePixels

    def getTotalCircles(self):
        '''
        Get Total circles for current in processing image
        '''
        return self.totalCircles

    def getTotalOilPixels(self):
        '''
        Get Total oil pixels for current in processing image
        '''
        return self.totalOilPixels

    def getTotalPlatePixels(self):
        '''
        Get Total plate pixels for current in processing image
        '''
        return self.totalPlatePixels

    def getTotalPixels(self):
        '''
        Get Total image pixels for current in processing image
        '''
        return self.totalPixels

    def saveImageAs(self, filePath, name, image):
        '''
        Save image at selected filepath
        '''
        Path(filePath).mkdir(parents=True, exist_ok=True)
        os.chdir(filePath)
        cv2.imwrite(name, image)

    def processFiles(self):
        '''
        Process all files with the selected configuration
        '''
        # Get current working directory when running function
        cwd = os.getcwd()
        resultsBasePath = os.path.join(cwd, 'results')
        # Check if directory not exist
        Path(resultsBasePath).mkdir(parents=True, exist_ok=True)

        files = self.getFilesInPath()

        if len(files) > 0:
            header = ['#', 'File Path', 'Total Pixels', 'Total Circles',
                      'Total Circles pixels', 'Total Oil pixels', 'Total plate pixels', 'Oil / Plate percent']
            timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

            # Start generating csv file
            with open("{basePath}/results_{date}.csv".format(basePath=resultsBasePath, date=timestr), 'w', encoding='UTF8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                # Start Showing progress bar
                with progressbar.ProgressBar(max_value=len(files)) as bar:
                    # Enumrate founded files
                    for index, filePath in enumerate(files, start=1):
                        fileDir = os.path.dirname(filePath)
                        fileName = os.path.basename(filePath)

                        img = self.getImageAtPath(filePath)

                        # Check if showFilePreview is True show plot
                        if self.showFilePreview:
                            plt.subplot(151), plt.imshow(img), plt.title(
                                'Original Image'), plt.xticks([]), plt.yticks([])

                        img = self.maskCirclsImage(img)
                        img = self.maskedOilImage(img)
                        img = self.maskedPlateImage(img)

                        # Save converted image file
                        self.saveImageAs(os.path.join(
                            resultsBasePath,
                            'images',
                            os.path.basename(
                                fileDir),
                        ),
                            fileName, img)

                        # write CSV file record for current processed image
                        writer.writerow({
                            '#': index,
                            'File Path': filePath,
                            'Total Pixels': self.getTotalPixels(),
                            'Total Circles': self.getTotalCircles(),
                            'Total Circles pixels': self.getTotalCirclePixels(),
                            'Total Oil pixels': self.getTotalOilPixels(),
                            'Total plate pixels': self.getTotalPlatePixels(),
                            'Oil / Plate percent': (
                                (self.getTotalOilPixels() / (self.getTotalPlatePixels() + self.getTotalOilPixels())
                                 ) * 100
                            )
                        })

                        # Check if showFilePreview is True show plot
                        if self.showFilePreview:
                            plt.subplot(152), plt.imshow(img[:, :, 2], cmap='gray'), plt.title(
                                'Circle'), plt.xticks([]), plt.yticks([])
                            plt.subplot(153), plt.imshow(img[:, :, 0], cmap='gray'), plt.title(
                                'Oil'), plt.xticks([]), plt.yticks([])
                            plt.subplot(154), plt.imshow(img[:, :, 1], cmap='gray'), plt.title(
                                'plate'), plt.xticks([]), plt.yticks([])
                            plt.subplot(155), plt.imshow(img), plt.title(
                                'Final Image'), plt.xticks([]), plt.yticks([])
                            plt.show()
                        bar.update(index)
