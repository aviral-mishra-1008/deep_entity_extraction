'''
Install dependencies: pip install easyocr
'''

class Image2Text:
    import easyocr as es
    def load_reader(self):
        reader = self.es.Reader(['en'])
        return reader
    
    def extract_text(self,reader,path):
        reader = self.load_reader()
        extracts = reader.readtext(path)

        result = []

        for detection in extracts:
            result.append(detection[1])
        
        return result


'''
Instructions for use

from image_to_text import Image2Text

1. Declare a model using class Image2Text
>>> model = Image2Text()

2. Declare a reader using model, do this once to avoid reloads each time
>>> reader = model.load_reader()

3. Specify reader and path to image to get the extracted information
>>> result = model.extract_text(reader,__path_to_image__)

Result is a bag of words extracted from the image in form of a python <list> 
'''