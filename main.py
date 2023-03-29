"""
Código para probar la librería de face_recognition.
Link: https://face-recognition.readthedocs.io/en/latest/face_recognition.html
Esta libería nos permite,
* Localizar caras humanas en una imagen.
* Reconocer caras previamente definidas.
* Trabajar con landmarks.
* Saber diferencia de similitud entre las diferentes caras.
"""

import os
import cv2
import face_recognition
     
def train(paths=None, model='hog'):
    """
    Función en la que entrenamos el reconocedor de caras
    Inputs:
    * paths: (tupla) Rutas de trabajo. Si [0] es None activamos la cam.
    * model: (str) face_recognition permite emplear los modelos 'hog' (por defecto)
            y 'cnn' para llevar a cabo la localización de las caras.
    """
    # Obtenemos rutas de trabajo
    path_in, path_out = paths
    # Trabajamos cam/foto según defina usuario
    if path_in is None:
        # Activamos cam
        cap = cv2.VideoCapture(0)
        while True:
            # Activamos y volteamos para evitar efecto espejo
            ret, frame = cap.read()
            if ret == False: break
            frame = cv2.flip(frame, 1)
            cv2.imshow("Train", frame)
            # Comprobamos si usuario presiona tecla espacio para hacer foto
            k = cv2.waitKey(1)
            if k == 32 & 0xFF: # esp
                # Aplicamos reconocimento facial
                face_loc = face_recognition.face_locations(frame, model=model)[0]
                face_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_loc])[0]
                # Dibujamos bbox y guardamos la imagen de train
                cv2.rectangle(frame, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 255, 0))
                cv2.imwrite(path_out, frame)
                break
        # Liberamos recursos y cerramos ventanas
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Leemos la imagen y aplicamos reconocimiento facial
        img = cv2.imread(path_in)
        # Aplicamos reconocimento facial
        face_loc = face_recognition.face_locations(img)[0]
        face_encodings = face_recognition.face_encodings(img, known_face_locations=[face_loc])[0]
        # Dibujamos bbox y guardamos la imagen de train
        cv2.rectangle(img, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (0, 255, 0))
        cv2.imwrite(path_out, img)
        
    return face_encodings


def test(face_image_encodings, paths, model='hog'):
    """
    Función para reconocer rostros.
    Imputs:
        * face_image_encodings: (var) Valores que definen el rostro que queremos reconocer.
        * paths: (tupla) Rutas de trabajo. Si [0] es None activamos la cam.
        * model: (str) face_recognition permite emplear los modelos 'hog' (por defecto)
            y 'cnn' para llevar a cabo la localización de las caras.
    """
    # Obtenemos rutas de trabajo
    path_in, path_out = paths
    # Trabajamos cam/foto según defina usuario
    if path_in is None:
        # Activamos cámara
        cap = cv2.VideoCapture(0)
        while True:
            # Activamos y volteamos para evitar efecto espejo
            ret, frame = cap.read()
            if ret == False: break
            frame = cv2.flip(frame, 1)
            cv2.imshow("Test", frame)
            # Comprobamos si usuario presiona tecla espacio para hacer foto
            k = cv2.waitKey(1)
            if k == 32 & 0xFF: # esp
                for face_location in face_recognition.face_locations(frame):
                    face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
                    result = face_recognition.compare_faces([face_image_encodings], face_frame_encodings)
                    if result[0] == True:
                            text, color = "True",(125, 220, 0)
                    else:
                            text, color = "False", (50, 50, 255)

                    cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color, -1)
                    cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
                    cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

                cv2.imwrite(path_out, frame)
                break
        # Liberamos recursos y cerramos ventanas
        cap.release()
        cv2.destroyAllWindows()
    else:
         # Leemos la imagen y aplicamos reconocimiento facial
        img = cv2.imread(path_in)
        for face_location in face_recognition.face_locations(img, model=model):
            face_frame_encodings = face_recognition.face_encodings(img, known_face_locations=[face_location])[0]
            result = face_recognition.compare_faces([face_image_encodings], face_frame_encodings)
            if result[0] == True:
                    text, color = "True",(125, 220, 0)
            else:
                    text, color = "False", (50, 50, 255)

            cv2.rectangle(img, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color, -1)
            cv2.rectangle(img, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
            cv2.putText(img, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

        cv2.imwrite(path_out, img)

if __name__ == "__main__":

    # Defimos las rutas de trabajo
    current_dir = os.getcwd()
    file_name_train = 'MichaelScott.png'
    train_path_in = os.path.join(current_dir, 'myFiles', file_name_train)
    train_path_out = os.path.join(current_dir, 'results', 'train.png')
    file_name_test = ['all.png', 'all2.png', 'dwight.png']
    test_path_in_list = [os.path.join(current_dir, 'myFiles', file_name) for file_name in file_name_test]
    test_path_out_list = [os.path.join(current_dir, 'results', 'test'+str(i+1)+'.png') for i in range(len(file_name_test))]

    # SI QUEREMOS TRABAJAR CON LA CAM DESCOMENTAMOS SIGUIENTES LÍNEAS
    #train_path_in, test_path_in_list = None, [None]
    #train_path_out = os.path.join(current_dir, 'results', 'trainCAM.png')
    #test_path_out_list = [os.path.join(current_dir, 'results', 'testCAM.png')]

    res = train(paths = (train_path_in, train_path_out), model='hog')
    for test_path_in, test_path_out in zip(test_path_in_list, test_path_out_list):
        test(res, paths = (test_path_in, test_path_out), model='hog')

    print(' *** FIN PROGRAMA ***\n')