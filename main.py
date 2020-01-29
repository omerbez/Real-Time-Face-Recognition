import cv2
import numpy as np
import face_recognition as fr
import utils


"""
    Speed Optimizations:
    1) Process faces recognition once per N frames (improves little)
    2) Process faces recognition ONLY when faces amount has been changed.
       otherwise, don't run face recognition but track the faces which already detected before.
    ** in-order to improve reliability, despite the second condition, we try to
       recognize the face several times ("STABILIZATION_TIME") because we don't want
       to "get stuck" with "Unknown" when a face, for example, 
       couldn't be recognized at first (due to person position), but after further
       chances it will be recognized well. (so we try at list several times before applying condition 2) 
    3) Frame size scaled-down by 50% before being processed. (reduce detection distance but improves speed)
"""


video_capture = cv2.VideoCapture(0)

print("Loading faces from data folder, please wait...")
(known_face_encodings, known_face_names) = utils.loadFacesData()
print("Done!")

facesLocations = None
shownFacesLocations = []
faceEncodings = None
detectedNames = []
stabilizationCounter = 0
frameCounter = -1

while True:
    ret, frame = video_capture.read()
    frameCounter += 1

    # Resize frame of video to 1/2 size for faster detection and recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame = small_frame[:, :, ::-1]

    # Find all the faces in the frame..
    facesLocations = fr.face_locations(small_frame)

    # Process ONLY when all 3 of optimizations conditions are met!
    if (frameCounter % utils.N == 0) and (
            len(shownFacesLocations) != len(facesLocations) or stabilizationCounter < utils.STABILIZATION_TIME):

        # if it's "new" case (shown faces != current) check more frames for a stable result
        if len(shownFacesLocations) != len(facesLocations):
            stabilizationCounter = 0
        elif utils.alreadyRecognized(detectedNames):
            # if all the faces are recognized - we are "stabled" - no need for further processing.
            stabilizationCounter = utils.STABILIZATION_TIME
            continue
        else: stabilizationCounter += 1

        # Encode the detected faces.
        faceEncodings = fr.face_encodings(small_frame, facesLocations)

        # Remember the faces locations in order to show them at the next frames without re-processing
        shownFacesLocations = facesLocations
        detectedNames.clear()

        for faceEncode in faceEncodings:
            # Check every known face and return a boolean array (match or not.. default tolerance is 0.6)
            matches = np.asarray(fr.compare_faces(known_face_encodings, faceEncode, tolerance=0.6))
            name = utils.UNKNOWN_NAME

            # If we got more than one match - get the nearest one ("manually checking"..)
            trueMatches = matches[matches == True]
            if len(trueMatches) > 1:
                # Calculate the distance from each known-face to the detected face (check how similar they are..)
                distances = fr.face_distance(known_face_encodings, faceEncode)
                bestIndex = np.argmin(distances)  # Take the index of the minimum distance == best match..
                name = known_face_names[bestIndex]
            elif len(trueMatches) == 1:
                name = known_face_names[utils.findIndexOf(matches, True)]

            detectedNames.append(name)  # add the name.. may be "unKnown"..

    # if no face-encoding comparision occurred, update the shown faces locations "manually"
    # Beware - since we process face-encoding once a 10 frames, the len of "facesLocation" may be smaller/larger
    # than the shownLocations.. (if new face was added/removed..) so the function should care about that too
    if shownFacesLocations != facesLocations:
        shownFacesLocations = utils.calculateNextFacesLocations(shownFacesLocations, facesLocations)

    # draw rect on the frame around the detected faces including the names.
    utils.drawRectAndName(frame, shownFacesLocations, detectedNames)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
