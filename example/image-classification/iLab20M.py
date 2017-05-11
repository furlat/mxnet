# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:22:56 2016

@author: jiaping
"""

"""
iLab20M module
    parseFileName: input - fileName, output - dict
    genFileName: generate a standard iLab20M file Name
    readImage: read image from iLab20M database
"""
from numbers import Number

def genFileName(**args):
    # car-i0160-b0001-c00-r00-l0-f1.png
    args = validateNamingConvension(args);
    
    indList  = ['Instance', 'Background', 'Camera', 
                'Rotation', 'Light', 'Focus'];
    fileName = '';
    sep = '-';

    categories = getClasses();
    for ind in args:
        if args[ind] in categories:
            fileName += str(args[ind]);
            break;            
    
    for ind in indList:
        if ind in args:
            partName = idx2str({ind: int(args[ind])});
            fileName += sep + partName;
    return fileName;                
        
                    
    
    

def parseFileName(fileName):
    # a typical file name looks like:
    # car-i0160-b0001-c00-r00-l0-f1.png
    extIdx = fileName.index('.');
    if isinstance(extIdx, list):
        fileName = fileName[0:extIdx[-1]];
    elif isinstance(extIdx, Number):
        fileName = fileName[0:extIdx];        
    sep = '-';
    parts = fileName.split(sep);
    
    categories = getClasses();
    Class = categories.intersection(parts);
    if not Class:
        raise ValueError('the fileName is not \
                          in the convension of iLab20M');
    defaultNaming = getDefaultNamingConvension();
    defaultMapping = getDefaultNamingMapping();
    for p in parts:
        if p in Class:
            defaultNaming['Class'] = p;
        else:            
            key = p[0];
            val = p[1:];
            try:
                defaultNaming[defaultMapping[key]] = int(val)
            except:
                raise ValueError('the fileName is not \
                                 in the naming convension of iLab20M');
                                         
    return defaultNaming;
    
    
def getClasses():
    categories = {'car',  
                  'f1car',  
                  'bus',   
                  'semi', 
                  'tank',  
                  'plane', 
                  'pickup', 
                  'train', 
                  'mil',  
                  'heli'};    
    return categories;                  
    
def getDefaultNamingConvension():
    naming = {'Class': 'car',
              'Instance': 1,
              'Background': 1,
              'Camera': 1,
              'Rotation': 1,
              'Light': 1,
              'Focus':1}
    return naming
    
def getDefaultNamingMapping():
    mapping = {'': 'Class', 
               'i': 'Instance', 
               'b': 'Background', 
               'c': 'Camera', 
               'l': 'Light',
               'f': 'Focus',
               'r': 'Rotation'};
    return mapping;   


def idx2strInstance(idx):
    idx = 10000 + idx;
    idxstr = str(idx);
    name = 'i' + idxstr[1:];
    return name;

def idx2strBackground(idx):
    idx = 10000 + idx;
    idxstr = str(idx);
    name = 'b' + idxstr[1:];
    return name;

def idx2strCamera(idx):
    idx = 100 + idx;
    idxstr = str(idx);
    name = 'c' + idxstr[1:];
    return name;

def idx2strRotation(idx):
    idx = 100 + idx;
    idxstr = str(idx);
    name = 'r' + idxstr[1:];
    return name;

def idx2strLight(idx):
    idx = 10 + idx;
    idxstr = str(idx);
    name = 'l' + idxstr[1:];
    return name;

def idx2strFocus(idx):
    idx = 10 + idx;
    idxstr = str(idx);
    name = 'f' + idxstr[1:];
    return name;

def idx2str(dictIdx):
    if len(dictIdx) != 1:
        raise ValueError('the input should be a tuple');
    defaultNamingConvension = getDefaultNamingConvension();        
    for i in dictIdx:
        if i not in defaultNamingConvension:
            raise ValueError('the indicator is not consistent with the standard iLab20M\
                                naming convensions');
        else:
            if i == 'Instance':
                return idx2strInstance(dictIdx[i]);
            elif i == 'Background':
                return idx2strBackground(dictIdx[i]);
            elif i == 'Focus':
                return idx2strFocus(dictIdx[i]);
            elif i == 'Light':
                return idx2strLight(dictIdx[i]);
            elif i == 'Camera':
                return idx2strCamera(dictIdx[i]);
            elif i == 'Rotation':
                return idx2strRotation(dictIdx[i]);                
   
   

def validateNamingConvension(args):
    defaultName = getDefaultNamingConvension();
    for key in args:
        if key in defaultName:
            defaultName[key] = args[key]
        else:
            print ('%s is not a standard keyword\n') % key            
        
    return defaultName;
    
def validateNamingConvension2(**args):
    defaultName = getDefaultNamingConvension();
    for key in args:
        if key in defaultName:
            defaultName[key] = args[key]
        else:
            print ('%s is not a standard keyword\n') % key            
        
    return defaultName;    
    
if __name__ == '__main__':
    parsedFileName = parseFileName('car-i0160-b0001-c00-r00-l0-f1.png');
    fileNameParts = validateNamingConvension({'Class': 'bus', 'Instance': 10, 
                                        'Light': 0, 'Focus': 3});
    fileName = genFileName(Class = 'tank', Instance = 9, 
                           Rotation = 5, Camera = 8, Light = 1);

    ind = 'done';                                        
                  