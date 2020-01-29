from os import listdir
from os.path import isfile, join
import face_recognition as fr
import cv2
from classes import Point


# define const variables
UNKNOWN_NAME = "Unknown"  # the text of unrecognized faces..
FACE_PADDING = 5  # rect padding

# "Convergence time" - how many times we process face-recognition on same face
# before we "tag" it with it's result, and won't process it again.
STABILIZATION_TIME = 3
N = 2  # process 1 per N frames.


def findIndexOf(a, value):
    """
    Find the first index of "value" in ndarray "a"
    :param a: numpy ndarray to search in
    :param value: the desired value
    :return: The first index of "value", or -1 if not found.
    """
    for i in range(len(a)):
        if a[i] == value:
            return i

    return -1


def alreadyRecognized(detectedNames):
    """
    check if all the list items are recognized == their value is not "Unknown"
    :param detectedNames: list of names (str)
    :return: True if all the list's items are not "Unknown" value.
             False if at least one item like that exists.
    """
    for name in detectedNames:
        if name == UNKNOWN_NAME:
            return False

    return True


def getDistanceOf(rect1, rect2):
    """
    calculate the distance between rect1 and rect2
    :param rect1: tuple of 4 - (top, right, bottom, left)
    :param rect2: tuple of 4 - (top, right, bottom, left)
    :return: euclidean distance between upper-left point of the rects.
    """
    p1 = Point(rect1[3], rect1[0])  # upper-left point of rect1
    p2 = Point(rect2[3], rect2[0])  # upper-left point of rect2
    return p1.distanceFrom(p2)


def calculateNextFacesLocations(currentLocations, nextLocations):
    """
    match between every item x of "currentLocation" it's next location y of "nextLocations" list.
    in other words, figure-out which item in "nextLocations" is the next Location of item x in "currentLocations".
    Note: the lists len may be different. the function will handle that so some items will remain the same
    (won't have a match because there are closer items).
    :param currentLocations: list of the current faces location which the user currently sees - faces from last frames.
    :param nextLocations: list of the coming (next) frame's faces locations.
    :return: new list of faces to show. the list size will be the same as "currentLocations" list.
    """
    if len(nextLocations) == 0 or len(currentLocations) == 0:
        return currentLocations

    temp = currentLocations.copy()
    if len(currentLocations) >= len(nextLocations):
        for i in range(len(nextLocations)):
            minVal = -1
            minIndex = -1
            for j in range(len(currentLocations)):
                dist = getDistanceOf(currentLocations[j], nextLocations[i])
                if minVal == -1 or dist < minVal:
                    minVal = dist
                    minIndex = j
            temp[minIndex] = nextLocations[i]
    else:
        for i in range(len(currentLocations)):
            minVal = -1
            minIndex = -1
            for j in range(len(nextLocations)):
                dist = getDistanceOf(nextLocations[j], currentLocations[i])
                if minVal == -1 or dist < minVal:
                    minVal = dist
                    minIndex = j
            temp[i] = nextLocations[minIndex]

    return temp


def loadFacesData():
    """
    load faces data and names from data folder.
    :return: a tuple of 2 lists - the encoded faces & names list.
    """
    path = "data"
    filesList = [f for f in listdir(path) if isfile(join(path, f))]
    faces = []
    names = []
    for f in filesList:
        img = fr.load_image_file(join(path, f))
        faces.append(fr.face_encodings(img)[0])
        dotIndex = f.find(".")
        names.append(f[0:(dotIndex if dotIndex != -1 else f.__len__())])

    return faces, names


def drawRectAndName(frame, shownFacesLocations, detectedNames):
    for (top, right, bottom, left), name in zip(shownFacesLocations, detectedNames):
        # Scale back up face locations because the detected face was scaled-down..
        top = (top*2) - FACE_PADDING
        right = (right*2) + FACE_PADDING
        bottom = (bottom*2) + FACE_PADDING
        left = (left*2) - FACE_PADDING

        # Draw a box around the face
        rectColor = (0, 0, 255) if name == UNKNOWN_NAME else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color=rectColor, thickness=2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom-20), (right, bottom), rectColor, cv2.FILLED)
        cv2.putText(frame, name, (left+FACE_PADDING, bottom-FACE_PADDING),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
