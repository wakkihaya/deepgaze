#Import required libraries
import cv2
import numpy as np
from deepgaze.color_classification import HistogramColorClassifier
import unittest

class TestHistClassifier(unittest.TestCase):
          """Test cases to ensure color classification with Histogram Color Classifier works as expected"""
                
          def __init__(self,*args, **kwargs):
              
              
                super(TestHistClassifier, self).__init__(*args, **kwargs)
                
                self.classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128], hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')
    
                BASE_PATH="Data/Color_Classification/model_"
                models=[]
                self.names=["Flash","Batman","Hulk","Superman","Captain America","Wonder Woman","Iron Man","Wolverine"]
          
                for i in range(1,9):
    
                      models.append(cv2.imread(BASE_PATH+str(i)+'.png'))
    
                for model,name in zip(models,self.names):
     
                      self.classifier.addModelHistogram(model,name=name)
                      
                self.test_image=cv2.imread('Data/Color_Classification/image_2.jpg')
              
    
          def test_ComparisonArray(self):
    
                expected_comparison=[0.00818883,0.55411926,0.12405966,0.07735263,0.34388389,0.12672027,0.09870308,0.2225694 ]
                #expected_name=['Flash', 'Batman', 'Hulk', 'Superman', 'Captain America', 'Wonder Woman', 'Iron Man', 'Wolverine']
    
                comparison_array = self.classifier.returnHistogramComparisonArray(self.test_image, method="intersection")
        
                assert np.allclose(comparison_array,expected_comparison)
                
          def test_Names(self):
             
                expected_name=['Flash', 'Batman', 'Hulk', 'Superman', 'Captain America', 'Wonder Woman', 'Iron Man', 'Wolverine']
                
                assert self.classifier.returnNameList()==expected_name
                
          def test_BestMatchIndex(self):
              
                assert self.classifier.returnBestMatchIndex(self.test_image)==1
                
          def test_BestMatchName(self):
              
                assert self.classifier.returnBestMatchName(self.test_image)=='Batman'
                
          def test_Cache(self):
              
                expected_name=['Flash', 'Batman', 'Hulk', 'Superman', 'Captain America', 'Wonder Woman', 'Iron Man', 'Wolverine']
                
                self.classifier.returnHistogramComparisonArray(self.test_image, method="intersection")
                assert self.classifier.returnNameList()==expected_name
                assert self.classifier.returnBestMatchIndex()==1
                assert self.classifier.returnBestMatchName()=='Batman'
            
    
if __name__=='__main__':
    
    unittest.main()
    
    
    
    
    

