#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import cv2
import sys
import warnings

(MAJOR, MINOR, _) = cv2.__version__.split('.')
VERSION_ALERT = '[DEEPGAZE][ERROR] color_classification.py: the version ' + MAJOR + ' of OpenCV is not compatible with Deepgaze 2.0'

class HistogramColorClassifier:
    """Classifier for comparing an image I with a model M. The comparison is based on color
    histograms. It included an implementation of the Histogram Intersection algorithm.

    The histogram intersection was proposed by Michael Swain and Dana Ballard 
    in their paper "Indexing via color histograms".
    Abstract: The color spectrum of multicolored objects provides a a robust, 
    efficient cue for indexing into a large database of models. This paper shows 
    color histograms to be stable object representations over change in view, and 
    demonstrates they can differentiate among a large number of objects. It introduces 
    a technique called Histogram Intersection for matching model and image histograms 
    and a fast incremental version of Histogram Intersection that allows real-time 
    indexing into a large database of stored models using standard vision hardware. 
    Color can also be used to search for the location of an object. An algorithm 
    called Histogram Backprojection performs this task efficiently in crowded scenes.
    """

    def __init__(self, Data=None,Names=None,channels=[0, 1, 2], hist_size=[10, 10, 10], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR'):
        """Init the classifier.

        This class has an internal list containing all the models.
        it is possible to append new models. Using the default values
        it extracts a 3D BGR color histogram from the image, using
	10 bins per channel.
        @param Data [List:Images]: Create an instance of classifier loaded with array of images
        @param Names [List:Names]: Names of initialized Image models
        @param channels [List:int]: where we specify the index of the channel 
           we want to compute a histogram for. For a grayscale image, 
           the list would be [0]. For all three (red, green, blue) channels, 
           the channels list would be [0, 1, 2].
        @param hist_size [List:int] number of bins we want to use when computing a histogram. 
            It is a list (one value for each channel). Note: the bin sizes can 
            be different for each channel.
        @param hist_range [List:int] it is the min-max value of the values stored in the histogram.
            For three channels can be [0, 256, 0, 256, 0, 256], if there is only one
            channel can be [0, 256]
        @param hsv_type [string] Convert the input BGR frame in HSV or GRAYSCALE. before taking 
            the histogram. The HSV representation can get more reliable results in 
            situations where light have a strong influence.
            BGR: (default) do not convert the input frame
            HSV: convert in HSV represantation
            GRAY: convert in grayscale
        """
        self.channels = channels
        self.hist_size = hist_size
        self.hist_range = hist_range
        self.hist_type = hist_type
        self.model_list = list()
        self.name_list = list()
        
        #Values cached after a comparison, only latest comparison values are cached and previous values are overwritten
        self.comparison=False
        self.best_match_name=" "
        self.best_match_index=None
        self.probability_array=[]
        self.comparison_array=[]
        
        #Initialize model with images and names specified
        self.InitializeModel(Data,Names)
        
        
    def InitializeModel(self,Images,Names):
        """This method allows HistogramColorClassifier instance to be loaded with required images and names.
           @param Images: Images provided as argument with histogram color classifier instance
           @param Names: Names provided as argument with histogram color classifier instance
        """
        
        #Assign images and names initialized with instance if specified
        if(Images is not None):
            
            #Check if names are assigned
            if(Names is not None):
                
                #Size of name array and Data must be equal
                if(len(Names)==len(Images)):
                    
                    for i in range(0,len(Images)):
                        
                        self.addModelHistogram(Images[i],name=Names[i])
                    
                else:
                    
                    warnings.warn("[DEEPGAZE][WARNING] The size of Name array and Data array do not match , ignoring Names of Images")
        
        
    def addModelHistogram(self, model_frame, name=''):
        """Add the histogram to internal container. If the name of the object
           is already present then replace that histogram with a new one.

        @param model_frame the frame to add to the model, its histogram
            is obtained and saved in internal list.
        @param name a string representing the name of the model.
            If nothing is specified then the name will be the index of the element.
        """
        if(self.hist_type=='HSV'): model_frame = cv2.cvtColor(model_frame, cv2.COLOR_BGR2HSV)
        elif(self.hist_type=='GRAY'): model_frame = cv2.cvtColor(model_frame, cv2.COLOR_BGR2GRAY)
        elif(self.hist_type=='RGB'): model_frame = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)
        elif(self.hist_type!='BGR'):            
            warnings.warn("[DEEPGAZE][ERROR] Please specify valid histogram type") 
            raise NameError
            
        hist = cv2.calcHist([model_frame], self.channels, None, self.hist_size, self.hist_range)
        hist = cv2.normalize(hist, hist).flatten()
        if name == '': name = str(len(self.model_list))
        if name not in self.name_list:
            self.model_list.append(hist)
            self.name_list.append(name)
        else:
            for i in range(len(self.name_list)):
                warnings.warn("[DEEPGAZE][WARNING] The given name " + name + 
                              " has been used before , it is overwriting previous instace")
                if self.name_list[i] == name:
                    self.model_list[i] = hist
                    break

    def removeModelHistogramByName(self, name):
        """Remove the specific model using the name as index.

        @param: name the index of the element to remove
        @return: True if the object has been deleted, otherwise False.
        """
        if name not in self.name_list:
            return False
        for i in range(len(self.name_list)):
            if self.name_list[i] == name:
                del self.name_list[i]
                del self.model_list[i]
                return True

    def returnHistogramComparison(self, hist_1, hist_2, method='intersection'):
        """Return the comparison value of two histograms.

        Comparing an histogram with itself return 1.
        @param hist_1
        @param hist_2
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        """
        if(method=="intersection"):
            if(MAJOR=='2'): flag = cv2.cv.CV_COMP_INTERSECT
            elif(MAJOR=='3'): flag = cv2.HISTCMP_INTERSECT
            else: raise ValueError(VERSION_ALERT)
            comparison = cv2.compareHist(hist_1, hist_2, flag)
        elif(method=="correlation"):
            if(MAJOR=='2'): flag = cv2.cv.CV_COMP_CORREL
            elif(MAJOR=='3'): flag = cv2.HISTCMP_CORREL
            else: raise ValueError(VERSION_ALERT)
            comparison = cv2.compareHist(hist_1, hist_2, flag)
        elif(method=="chisqr"):
            if(MAJOR=='2'): flag = cv2.cv.CV_COMP_CHISQR
            elif(MAJOR=='3'): flag = cv2.HISTCMP_CHISQR
            else: raise ValueError(VERSION_ALERT)
            comparison = cv2.compareHist(hist_1, hist_2, flag)
        elif(method=="bhattacharyya"):
            if(MAJOR=='2'): flag = cv2.cv.CV_COMP_BHATTACHARYYA
            elif(MAJOR=='3'): flag = cv2.HISTCMP_BHATTACHARYYA
            else: raise ValueError(VERSION_ALERT)
            comparison = cv2.compareHist(hist_1, hist_2, flag)
        else:
            raise ValueError('[DEEPGAZE][ERROR] color_classification.py: the method specified ' + str(method) + ' is not supported.')
        return comparison

    def returnHistogramComparisonArray(self, image, method='intersection',output_type=list):
        """Return the comparison array between all the model and the input image.

        The highest value represents the best match.
        @param image the image to compare
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        @return a numpy array containg the comparison value between each pair image-model
        """
        if(self.hist_type=='HSV'): image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif(self.hist_type=='GRAY'): image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif(self.hist_type=='RGB'): image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        comparison_array = np.zeros(len(self.model_list))
        image_hist = cv2.calcHist([image], self.channels, None, self.hist_size, self.hist_range)
        image_hist = cv2.normalize(image_hist, image_hist).flatten()
        counter = 0
        for model_hist in self.model_list:
            comparison_array[counter] = self.returnHistogramComparison(image_hist, model_hist, method=method)
            counter += 1
            
        self.comparison=True
        self.probability_array=np.divide(comparison_array, np.sum(comparison_array))
        self.best_match_index=np.argmax(comparison_array)
        self.best_match_name=self.name_list[self.best_match_index]
        self.comparison_array=comparison_array
        
        if(output_type==dict):
            
            comparison_array=dict(zip(self.name_list,comparison_array))
        
        return comparison_array

    def returnHistogramComparisonProbability(self, image=None, method='intersection',output_type=list):
        """Return the probability distribution of the comparison between 
        all the model and the input image. The sum of the elements in the output
        array sum up to 1.

        The highest value represents the best match.
        @param image the image to compare
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        @return a numpy array containg the comparison value between each pair image-model
        """
        comparison_distribution=None
        
        #Check if comparison is performed to use cached values
        if(self.comparison):
            
            comparison_distribution=self.probability_array
            
            if(output_type==dict):
                
                comparison_distribution=dict(zip(self.name_list,comparison_distribution))
            
        elif(image is not None):
            
           comparison_array = self.returnHistogramComparisonArray(image=image, method=method)
           #comparison_array[comparison_array < 0] = 0 #Remove negative values
           comparison_distribution = np.divide(comparison_array, np.sum(comparison_array))
           
        else:
            
           warnings.warn("[DEEPGAZE][WARNING] Classifier did not recieve an image instance")
            
            
        return comparison_distribution

    def returnBestMatchIndex(self, image=None, method='intersection'):
        """Return the index of the best match between the image and the internal models.

        @param image the image to compare
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        @return a numpy array containg the comparison value between each pair image-model
        """
        Index=None
        
        #Check if comparison is performed to use cached values
        if(self.comparison):
            
            Index=self.best_match_index
            
        elif(image is not None):
            
            comparison_array = self.returnHistogramComparisonArray(image, method=method)
            Index=np.argmax(comparison_array)
            
        else:
            
            warnings.warn("[DEEPGAZE][WARNING] Classifier did not recieve an image instance")
            
        return Index

    def returnBestMatchName(self, image=None, method='intersection'):
        """Return the name of the best match between the image and the internal models.

        @param image the image to compare
        @param method the comparison method.
            intersection: (default) the histogram intersection (Swain, Ballard)
        @return a string representing the name of the best matching model
        """
        name=None
        
        #Check if comparison is performed to use cached values
        if(self.comparison):
            name=self.best_match_name
        elif(image is not None):
            comparison_array = self.returnHistogramComparisonArray(image, method=method)
            arg_max = np.argmax(comparison_array)
            name=self.name_list[arg_max]
        else:
            warnings.warn("[DEEPGAZE][WARNING] Classifier did not recieve an image instance")
        return name

    def returnNameList(self):
        """Return a list containing all the names stored in the model.

        @return: a list containing the name of the models.
        """
        return self.name_list

    def returnSize(self):
        """Return the number of elements stored.

        @return: an integer representing the number of elements stored
        """
        return len(self.model_list)
