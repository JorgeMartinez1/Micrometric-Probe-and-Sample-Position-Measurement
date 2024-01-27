#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase para el manejo de los tags de Realidad Aumentada

Created on Tue Aug 9 14:16:53 2022
@author: jolumartinez
"""

import cv2
import numpy as np
import copy
import core.gestionKalman as gestionKalman
import time

def id_decode(image):  # To detect the ID information for the tag
    # ret, img_bw = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    ret3, img_bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corner_pixel = 255
    cropped_img = img_bw[50:150, 50:150]

    (h, w) = cropped_img.shape
    # calculate the center of the image
    center = (w / 2, h / 2)
    # print (h,w)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    found = False
    cv2.imshow("Tag after homogenous", cropped_img)
    block_1 = cropped_img[37, 37]
    block_3 = cropped_img[62, 37]
    block_2 = cropped_img[37, 62]
    block_4 = cropped_img[62, 62]
    white = 255
    if block_3 == white:
        block_3 = 1
    else:
        block_3 = 0
    if block_4 == white:
        block_4 = 1
    else:
        block_4 = 0
    if block_2 == white:
        block_2 = 1
    else:
        block_2 = 0
    if block_1 == white:
        block_1 = 1
    else:
        block_1 = 0
    # To get the orientation of the tag
    if cropped_img[85, 85] == corner_pixel:
        return list([block_4, block_3, block_2, block_1]), "BR"
    elif cropped_img[15, 85] == corner_pixel:
        return list([block_2, block_4, block_1, block_3]), "TR"
    elif cropped_img[15, 15] == corner_pixel:
        return list([block_1, block_2, block_3, block_4]), "TL"
    elif cropped_img[85, 15] == corner_pixel:
        return list([block_3, block_1, block_4, block_2]), "BL"

    return None, None


def get_code_location(image):
    ret3, img_bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corner_pixel = 255
    cropped_img = img_bw[50:150, 50:150]

    (h, w) = cropped_img.shape
    # calculate the center of the image
    center = (w / 2, h / 2)
    # print (h,w)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    found = False
    # cv2.imshow("Tag after homogenous", cropped_img)
    block_1 = cropped_img[37, 37]
    block_3 = cropped_img[62, 37]
    block_2 = cropped_img[37, 62]
    block_4 = cropped_img[62, 62]
    white = 255
    if block_3 == white:
        block_3 = 1
    else:
        block_3 = 0
    if block_4 == white:
        block_4 = 1
    else:
        block_4 = 0
    if block_2 == white:
        block_2 = 1
    else:
        block_2 = 0
    if block_1 == white:
        block_1 = 1
    else:
        block_1 = 0
    # To get the orientation of the tag
    if cropped_img[85, 85] == corner_pixel:
        return str(block_4)+str(block_3)+str(block_2)+str(block_1), "BR"
    elif cropped_img[15, 85] == corner_pixel:
        return str(block_2)+str(block_4)+str(block_1)+str(block_3), "TR"
    elif cropped_img[15, 15] == corner_pixel:
        return str(block_1)+str(block_2)+str(block_3)+str(block_4), "TL"
    elif cropped_img[85, 15] == corner_pixel:
        return str(block_3)+str(block_1)+str(block_4)+str(block_2), "BL"

    return None, None


def draw_cube(img, imgpts):  # To draw the cube
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 255), 3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 0), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def order(pts):  # To get the ordered points in a clockwise direction
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # print(np.argmax(diff))
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return ordered


def homo(p, p1):
    A = []
    p2 = order(p)
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    l = V[-1, :] / V[-1, -1]
    h = np.reshape(l, (3, 3))
    return h


def contour_generator(frame):
    test_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    test_blur = cv2.GaussianBlur(test_img1, (1, 1), 0)
    edge = cv2.Canny(test_blur, 60, 180)
    edge1 = copy.copy(edge)
    countour_list = list()
    ctrs, h = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(ctrs)
    # print(h[0][3])
    index = list()

    for hier in h[0]:
        if hier[3] != -1:
            index.append(hier[3])
            #print(hier[3])

    for c in index:
        peri = cv2.arcLength(ctrs[c], True)
        approx = cv2.approxPolyDP(ctrs[c], 0.005 * peri, True)
        # print(len(ctrs[c-1]))

        if len(approx) >= 4:
            peri1 = cv2.arcLength(ctrs[c], True)
            corners = cv2.approxPolyDP(ctrs[c], 0.015 * peri1, True)
            countour_list.append(corners)

    new_contour_list = list()
    for contour in countour_list:
        if len(contour) == 4:
            new_contour_list.append(contour)
    final_contour_list = list()
    for element in new_contour_list:
        if 1000 < cv2.contourArea(element) < 10000:
            final_contour_list.append(element)

    return final_contour_list


def reorient(location, maxDim):
    if location == "BR":
        p1 = np.array([
            [0, 0],
            [maxDim - 1, 0],
            [maxDim - 1, maxDim - 1],
            [0, maxDim - 1]], dtype="float32")
        return p1
    elif location == "TR":
        p1 = np.array([
            [maxDim - 1, 0],
            [maxDim - 1, maxDim - 1],
            [0, maxDim - 1],
            [0, 0]], dtype="float32")
        return p1
    elif location == "TL":
        p1 = np.array([
            [maxDim - 1, maxDim - 1],
            [0, maxDim - 1],
            [0, 0],
            [maxDim - 1, 0]], dtype="float32")
        return p1

    elif location == "BL":
        p1 = np.array([
            [0, maxDim - 1],
            [0, 0],
            [maxDim - 1, 0],
            [maxDim - 1, maxDim - 1]], dtype="float32")
        return p1


def image_process(frame, p1):
    frame_work = frame.copy()
    final_contour_list = contour_generator(frame_work)

    for i in range(len(final_contour_list)):
        # print("Contour: ", final_contour_list[i])
        cv2.drawContours(frame_work, [final_contour_list[i]], -1, (0, 255, 0), 1)

        c_rez = final_contour_list[i][:, 0]
        H_matrix = homo(p1, order(c_rez))
        tag = cv2.warpPerspective(frame_work, H_matrix, (200, 200))

        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        decoded, location = id_decode(tag1)

        cv2.putText(frame_work, str(decoded)+str(location), final_contour_list[i][0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA, False)

    return frame_work


def get_all_puntos_tags(tags_collection, tags_collection_patron):
    puntos_frame = []
    puntos_patron = []
    todos_los_puntos = []
    puntos_3d = []
    for key, value in tags_collection.items():
        for key2, value2 in tags_collection_patron.items():
            if key == key2 != '15BL' and value['tipo'] != 'inicial' and value2['tipo'] != 'inicial':
                puntos_frame.append(((value['corners'][(0)]), value['corners'][(1)]))
                puntos_frame.append(((value['corners'][(2)]), value['corners'][(3)]))
                puntos_frame.append(((value['corners'][(4)]), value['corners'][(5)]))
                puntos_frame.append(((value['corners'][(6)]), value['corners'][(7)]))

                puntos_patron.append(((value2['corners'][(0)]), value2['corners'][(1)]))
                puntos_patron.append(((value2['corners'][(2)]), value2['corners'][(3)]))
                puntos_patron.append(((value2['corners'][(4)]), value2['corners'][(5)]))
                puntos_patron.append(((value2['corners'][(6)]), value2['corners'][(7)]))

    if len(puntos_patron) == len(puntos_frame) >= 4:
        (H, status) = cv2.findHomography(np.array(puntos_patron), np.array(puntos_frame), 0, 3)
        for key, value in tags_collection_patron.items():

            puntos_3d.append(((value['corners'][(0)]), value['corners'][(1)], 0))
            puntos_3d.append(((value['corners'][(2)]), value['corners'][(3)], 0))
            puntos_3d.append(((value['corners'][(4)]), value['corners'][(5)], 0))
            puntos_3d.append(((value['corners'][(6)]), value['corners'][(7)], 0))

            new_pts_1 = np.float32([[value['corners'][(0)], value['corners'][(1)]], [value['corners'][(2)],\
                                    value['corners'][(3)]], [value['corners'][(4)], value['corners'][(5)]],\
                                    [value['corners'][(6)], value['corners'][(7)]]]).reshape(-1, 1, 2)

            transformados = cv2.perspectiveTransform(new_pts_1, H)

            todos_los_puntos.append(((transformados[0][0][0]), transformados[0][0][1]))
            todos_los_puntos.append(((transformados[1][0][0]), transformados[1][0][1]))
            todos_los_puntos.append(((transformados[2][0][0]), transformados[2][0][1]))
            todos_los_puntos.append(((transformados[3][0][0]), transformados[3][0][1]))

    """
    todos_los_puntos.append(((tags_collection[15]['corners'][(0)]), tags_collection[15]['corners'][(1)]))
    todos_los_puntos.append(((tags_collection[15]['corners'][(2)]), tags_collection[15]['corners'][(3)]))
    todos_los_puntos.append(((tags_collection[15]['corners'][(4)]), tags_collection[15]['corners'][(5)]))
    todos_los_puntos.append(((tags_collection[15]['corners'][(6)]), tags_collection[15]['corners'][(7)]))
    
    """

    return todos_los_puntos, puntos_3d


def ordered_points_list(tags_collection_izq, tags_collection_der):
    puntos_izq = []
    puntos_der = []
    puntos_3d = []
    for key, value in tags_collection_izq.items():
        if key != '15BL':
            #print("Llave: ", key)
            cornersIzq = tags_collection_izq[key]['corners']
            cornersDer = tags_collection_der[key]['corners']

            #print('Corners: ', cornersIzq)

            puntos_izq.append((cornersIzq[0]))
            puntos_izq.append((cornersIzq[1]))
            puntos_izq.append((cornersIzq[2]))
            puntos_izq.append((cornersIzq[3]))

            puntos_der.append((cornersDer[0]))
            puntos_der.append((cornersDer[1]))
            puntos_der.append((cornersDer[2]))
            puntos_der.append((cornersDer[3]))

            puntos_3d.append((tags_collection_izq[key]['corners_3D'][(0)]))
            puntos_3d.append((tags_collection_izq[key]['corners_3D'][(1)]))
            puntos_3d.append((tags_collection_izq[key]['corners_3D'][(2)]))
            puntos_3d.append((tags_collection_izq[key]['corners_3D'][(3)]))

    return puntos_izq, puntos_der, puntos_3d

def ordered_points_list_real(tags_collection_izq, tags_collection_der, tags_collection_patron):
    puntos_izq = []
    puntos_der = []
    puntos_3d = []
    for key, value in tags_collection_izq.items():
        if key != '15BL' and tags_collection_izq[key]['tipo'] == 'actual' and tags_collection_der[key]['tipo'] == 'actual':
            #print("Llave: ", key)
            cornersIzq = tags_collection_izq[key]['corners']
            cornersDer = tags_collection_der[key]['corners']

            #print('Corners: ', cornersIzq)

            puntos_izq.append((cornersIzq[0]))
            puntos_izq.append((cornersIzq[1]))
            puntos_izq.append((cornersIzq[2]))
            puntos_izq.append((cornersIzq[3]))

            puntos_der.append((cornersDer[0]))
            puntos_der.append((cornersDer[1]))
            puntos_der.append((cornersDer[2]))
            puntos_der.append((cornersDer[3]))

            puntos_3d.append((tags_collection_patron[key]['corners_3D'][(0)]))
            puntos_3d.append((tags_collection_patron[key]['corners_3D'][(1)]))
            puntos_3d.append((tags_collection_patron[key]['corners_3D'][(2)]))
            puntos_3d.append((tags_collection_patron[key]['corners_3D'][(3)]))

    return puntos_izq, puntos_der, puntos_3d


def ordered_points_list_one_camera(tags_collection_cam, tags_collection_patron):
    puntos_izq = []
    puntos_3d = []
    for key, value in tags_collection_cam.items():
        if key != '15BL' and tags_collection_cam[key]['tipo'] == 'actual':
            #print("Llave: ", key)
            cornersIzq = tags_collection_cam[key]['corners']

            #print('Corners: ', cornersIzq)

            puntos_izq.append((cornersIzq[0]))
            puntos_izq.append((cornersIzq[1]))
            puntos_izq.append((cornersIzq[2]))
            puntos_izq.append((cornersIzq[3]))

            puntos_3d.append((tags_collection_patron[key]['corners_3D'][(0)]))
            puntos_3d.append((tags_collection_patron[key]['corners_3D'][(1)]))
            puntos_3d.append((tags_collection_patron[key]['corners_3D'][(2)]))
            puntos_3d.append((tags_collection_patron[key]['corners_3D'][(3)]))

    return puntos_izq, puntos_3d


def get_H_from_tags(tags_collection, tags_collection_patron):
    puntos_frame = []
    puntos_patron = []
    listado = []

    for key, value in tags_collection.items():
        listado.clear()
        for key2, value2 in tags_collection_patron.items():
            if key == key2 and value['location'] == 'TL':
                listado = list(np.array(value['corners']).flatten())
                puntos_frame.append(((listado[(0)]), listado[(1)]))
                puntos_frame.append(((listado[(2)]), listado[(3)]))
                puntos_frame.append(((listado[(4)]), listado[(5)]))
                puntos_frame.append(((listado[(6)]), listado[(7)]))

                puntos_patron.append(((value2['corners'][(0)]), value2['corners'][(1)]))
                puntos_patron.append(((value2['corners'][(2)]), value2['corners'][(3)]))
                puntos_patron.append(((value2['corners'][(4)]), value2['corners'][(5)]))
                puntos_patron.append(((value2['corners'][(6)]), value2['corners'][(7)]))

    if len(puntos_patron) == len(puntos_frame) >= 4:
        (H, status) = cv2.findHomography(np.array(puntos_patron), np.array(puntos_frame), 0, 3)
        return H, status
    return None, None


def get_all_synthetic_points_tags(tags_collection, tags_collection_patron, synthetic_collection):
    puntos_frame = []
    puntos_patron = []
    todos_los_puntos = []
    H_TL = None
    H_BL = None

    # print("Se hace una búsqueda de coincidencias entre los tags encontrados y los tags patrón TL")
    for key, value in tags_collection.items():
        for key2, value2 in tags_collection_patron.items():
            if key == key2 and value['tipo'] == 'actual' and value2['tipo'] != 'inicial' and value['location'] == 'TL':

                puntos_frame.append(((value['corners'][(0)]), value['corners'][(1)]))
                puntos_frame.append(((value['corners'][(2)]), value['corners'][(3)]))
                puntos_frame.append(((value['corners'][(4)]), value['corners'][(5)]))
                puntos_frame.append(((value['corners'][(6)]), value['corners'][(7)]))

                puntos_patron.append(((value2['corners'][(0)]), value2['corners'][(1)]))
                puntos_patron.append(((value2['corners'][(2)]), value2['corners'][(3)]))
                puntos_patron.append(((value2['corners'][(4)]), value2['corners'][(5)]))
                puntos_patron.append(((value2['corners'][(6)]), value2['corners'][(7)]))

    if len(puntos_patron) == len(puntos_frame) >= 4:
        # print("Si los puntos encontrados son mayores a 4, se calcula la homografía con el patrón.")
        (H_TL, status) = cv2.findHomography(np.array(puntos_patron), np.array(puntos_frame), cv2.LMEDS, 0.3)
        for key, value in tags_collection_patron.items():
            if value['location'] == 'TL':
                todos_los_puntos.clear()
                new_pts_1 = np.float32([[value['corners'][(0)], value['corners'][(1)]],
                                        [value['corners'][(2)], value['corners'][(3)]],
                                        [value['corners'][(4)], value['corners'][(5)]],
                                        [value['corners'][(6)], value['corners'][(7)]]]).reshape(-1, 1, 2)

                transformados = cv2.perspectiveTransform(new_pts_1, H_TL)

                todos_los_puntos.append(((transformados[0][0][0]), transformados[0][0][1]))
                todos_los_puntos.append(((transformados[1][0][0]), transformados[1][0][1]))
                todos_los_puntos.append(((transformados[2][0][0]), transformados[2][0][1]))
                todos_los_puntos.append(((transformados[3][0][0]), transformados[3][0][1]))

                kalman = synthetic_collection[key]['kalman']
                measurement_array = synthetic_collection[key]['measurement_array']
                dt_array = synthetic_collection[key]['dt_array']
                lapso_estimaciones = time.time() - synthetic_collection[key]['tiempo_estimacion']

                pred_puntos_tag = gestionKalman.prediccion_kalman_con_correccion(kalman, measurement_array,
                                                                                 np.array(todos_los_puntos),
                                                                                 dt_array, lapso_estimaciones)
                synthetic_collection[key]['tiempo_estimacion']= time.time()

                synthetic_collection[key]['corners'] = np.array(pred_puntos_tag).reshape(-1, 2)
                synthetic_collection[key]['corners_3D'] = tags_collection_patron[key]['corners_3D']

    puntos_frame.clear()
    puntos_patron.clear()
    todos_los_puntos.clear()
    # print("Se hace una búsqueda de coincidencias entre los tags encontrados y los tags patrón BL")
    for key, value in tags_collection.items():
        for key2, value2 in tags_collection_patron.items():
            if key == key2 != '15BL' and value['tipo'] == 'actual' and value2['tipo'] != 'inicial' and value['location'] == 'BL':
                puntos_frame.append(((value['corners'][(0)]), value['corners'][(1)]))
                puntos_frame.append(((value['corners'][(2)]), value['corners'][(3)]))
                puntos_frame.append(((value['corners'][(4)]), value['corners'][(5)]))
                puntos_frame.append(((value['corners'][(6)]), value['corners'][(7)]))

                puntos_patron.append(((value2['corners'][(0)]), value2['corners'][(1)]))
                puntos_patron.append(((value2['corners'][(2)]), value2['corners'][(3)]))
                puntos_patron.append(((value2['corners'][(4)]), value2['corners'][(5)]))
                puntos_patron.append(((value2['corners'][(6)]), value2['corners'][(7)]))

    if len(puntos_patron) == len(puntos_frame) >= 4:
        (H_BL, status) = cv2.findHomography(np.array(puntos_patron), np.array(puntos_frame), 0, 1)
        for key, value in tags_collection_patron.items():
            if value['location'] == 'BL' and key != '15BL':
                todos_los_puntos.clear()
                new_pts_1 = np.float32([[value['corners'][(0)], value['corners'][(1)]],
                                        [value['corners'][(2)], value['corners'][(3)]],
                                        [value['corners'][(4)], value['corners'][(5)]],
                                        [value['corners'][(6)], value['corners'][(7)]]]).reshape(-1, 1, 2)

                transformados = cv2.perspectiveTransform(new_pts_1, H_BL)

                todos_los_puntos.append(((transformados[0][0][0]), transformados[0][0][1]))
                todos_los_puntos.append(((transformados[1][0][0]), transformados[1][0][1]))
                todos_los_puntos.append(((transformados[2][0][0]), transformados[2][0][1]))
                todos_los_puntos.append(((transformados[3][0][0]), transformados[3][0][1]))

                kalman = synthetic_collection[key]['kalman']
                measurement_array = synthetic_collection[key]['measurement_array']
                dt_array = synthetic_collection[key]['dt_array']
                lapso_estimaciones = time.time() - synthetic_collection[key]['tiempo_estimacion']

                pred_puntos_tag = gestionKalman.prediccion_kalman_con_correccion(kalman, measurement_array,
                                                                                 np.array(todos_los_puntos),
                                                                                 dt_array, lapso_estimaciones)
                synthetic_collection[key]['tiempo_estimacion'] = time.time()

                synthetic_collection[key]['corners'] = np.array(pred_puntos_tag).reshape(-1, 2)
                synthetic_collection[key]['corners_3D'] = tags_collection_patron[key]['corners_3D']
    # print("Se asigna a syntetic collection punta el mismo valor de tags collection")
    # print(f"tags_collection: {tags_collection['15BL']['corners']}")
    # print(f"tags_collection_patron: {tags_collection_patron['15BL']['corners']}")
    # print(f"synthetic_collection: {synthetic_collection['15BL']['corners']}")
    synthetic_collection['15BL']['corners'] = tags_collection['15BL']['corners']

    return synthetic_collection, H_TL, H_BL


def get_listado_tags(tags_collection_izq, tags_collection_der, synthetic=False):
    puntos_frame_izq = []
    puntos_frame_der = []

    if synthetic:
        for key, value in tags_collection_izq.items():
            for key2, value2 in tags_collection_der.items():

                if key == key2:
                    if key != '15BL':
                        # print(f"key 1: {key}; key 2: {key2}; tipo datos: {type(value['corners'])}; longitud datos: {len(value['corners'])}; datos: {value['corners']}")
                        puntos_frame_izq.append(np.array(value['corners']).flatten().reshape(-1, 2))
                        puntos_frame_der.append(np.array(value2['corners']).flatten().reshape(-1, 2))
    else:
        for key, value in tags_collection_izq.items():
            for key2, value2 in tags_collection_der.items():

                if key == key2 and value['tipo'] == value2['tipo'] != 'inicial':
                    if key != '15BL':
                        puntos_frame_izq.append(np.array(value['corners']).flatten().reshape(-1, 2))
                        puntos_frame_der.append(np.array(value2['corners']).flatten().reshape(-1, 2))


    return np.array(puntos_frame_izq).flatten().reshape(-1,2), np.array(puntos_frame_der).flatten().reshape(-1,2)


def get_coordenadas_tags_punta(tags_collection_izq, tags_collection_der):
    puntos_frame_izq = []
    puntos_frame_der = []

    puntos_frame_izq.append(((tags_collection_izq['15BL']['corners'][(0)]), tags_collection_izq['15BL']['corners'][(1)]))
    puntos_frame_izq.append(((tags_collection_izq['15BL']['corners'][(2)]), tags_collection_izq['15BL']['corners'][(3)]))
    puntos_frame_izq.append(((tags_collection_izq['15BL']['corners'][(4)]), tags_collection_izq['15BL']['corners'][(5)]))
    puntos_frame_izq.append(((tags_collection_izq['15BL']['corners'][(6)]), tags_collection_izq['15BL']['corners'][(7)]))

    puntos_frame_der.append(((tags_collection_der['15BL']['corners'][(0)]), tags_collection_der['15BL']['corners'][(1)]))
    puntos_frame_der.append(((tags_collection_der['15BL']['corners'][(2)]), tags_collection_der['15BL']['corners'][(3)]))
    puntos_frame_der.append(((tags_collection_der['15BL']['corners'][(4)]), tags_collection_der['15BL']['corners'][(5)]))
    puntos_frame_der.append(((tags_collection_der['15BL']['corners'][(6)]), tags_collection_der['15BL']['corners'][(7)]))

    return puntos_frame_izq, puntos_frame_der


def find_tags(frame, tagsCollection):
    dim = 200
    p1 = np.array([
        [0, 0],
        [dim - 1, 0],
        [dim - 1, dim - 1],
        [0, dim - 1]], dtype="float32")

    frame_work = frame.copy()
    final_contour_list = contour_generator(frame_work)
    code_location = list()

    collection_work = tagsCollection

    for i in range(len(final_contour_list)):

        c_rez = final_contour_list[i][:, 0]
        H_matrix = homo(p1, order(c_rez))
        tag = cv2.warpPerspective(frame_work, H_matrix, (200, 200))

        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        decoded, location = get_code_location(tag1)

        if decoded is not None:
            code_location.append((decoded, location, order(final_contour_list[i][:, 0])))

    for key, value in collection_work.items():
        for codigo, orientacion, coordenadas_tag in code_location:
            if str(int(codigo, 2))+orientacion == key:

                kalman = tagsCollection[key]['kalman']
                measurement_array = tagsCollection[key]['measurement_array']
                dt_array = tagsCollection[key]['dt_array']
                lapso_estimaciones = time.time() - tagsCollection[key]['tiempo_estimacion']

                pred_puntos_tag = gestionKalman.prediccion_kalman_con_correccion(kalman,
                                                                                 measurement_array, coordenadas_tag,
                                                                                 dt_array, lapso_estimaciones)

                tagsCollection[key]['code'] = codigo
                tagsCollection[key]['location'] = orientacion
                tagsCollection[key]['corners'] = pred_puntos_tag
                tagsCollection[key]['tiempo_estimacion'] = time.time()
                tagsCollection[key]['tipo'] = 'actual'

        if tagsCollection[key]['tipo'] != 'actual':
            tagsCollection[key]['tipo'] = 'estimado'

            kalman = value['kalman']
            measurement_array = value['measurement_array']
            dt_array = value['dt_array']
            lapso_estimaciones = time.time() - value['tiempo_estimacion']

            pred_puntos_tag = gestionKalman.prediccion_kalman_sin_correccion(kalman, measurement_array, dt_array,
                                                                             lapso_estimaciones)

            # tagsCollection[key]['code'] = codigo
            tagsCollection[key]['corners'] = pred_puntos_tag

            if lapso_estimaciones > 0.5:
                tagsCollection[key]['tipo'] = 'inicial'
                tagsCollection[key]['tiempo_estimacion'] = time.time()-0.5

        if tagsCollection[key]['tipo'] == 'actual':
            tagsCollection[key]['tipo'] = 'estimado'

    return tagsCollection


def find_tags_new(frame, tagsCollection):
    dim = 200
    p1 = np.array([
        [0, 0],
        [dim - 1, 0],
        [dim - 1, dim - 1],
        [0, dim - 1]], dtype="float32")

    frame_work = frame.copy()
    final_contour_list = contour_generator(frame_work)
    code_location = list()

    collection_work = tagsCollection

    for i in range(len(final_contour_list)):

        c_rez = final_contour_list[i][:, 0]
        H_matrix = homo(p1, order(c_rez))
        tag = cv2.warpPerspective(frame_work, H_matrix, (200, 200))

        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        decoded, location = get_code_location(tag1)

        if decoded is not None:
            code_location.append((decoded, location, order(final_contour_list[i][:, 0])))

    for key, value in collection_work.items():
        tagsCollection[key]['tipo'] = 'inicial'
        for codigo, orientacion, coordenadas_tag in code_location:
            if str(int(codigo, 2))+orientacion == key != '5BL':

                kalman = tagsCollection[key]['kalman']
                measurement_array = tagsCollection[key]['measurement_array']
                dt_array = tagsCollection[key]['dt_array']
                lapso_estimaciones = time.time() - tagsCollection[key]['tiempo_estimacion']

                pred_puntos_tag = gestionKalman.prediccion_kalman_con_correccion(kalman,
                                                                                 measurement_array, coordenadas_tag,
                                                                                 dt_array, lapso_estimaciones)

                tagsCollection[key]['code'] = codigo
                tagsCollection[key]['location'] = orientacion
                tagsCollection[key]['corners'] = pred_puntos_tag
                tagsCollection[key]['tiempo_estimacion'] = time.time()
                tagsCollection[key]['tipo'] = 'actual'

    return tagsCollection


def find_tags_no_kalman(frame, tagsCollection):
    dim = 200
    p1 = np.array([
        [0, 0],
        [dim - 1, 0],
        [dim - 1, dim - 1],
        [0, dim - 1]], dtype="float32")

    frame_work = frame.copy()

    for imagen in range(20):
        final_contour_list = contour_generator(frame_work)
        code_location = list()

        collection_work = tagsCollection

        for i in range(len(final_contour_list)):

            c_rez = final_contour_list[i][:, 0]
            H_matrix = homo(p1, order(c_rez))
            tag = cv2.warpPerspective(frame_work, H_matrix, (200, 200))

            tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
            decoded, location = get_code_location(tag1)

            if decoded is not None:
                code_location.append((decoded, location, order(final_contour_list[i][:, 0])))

        for key, value in collection_work.items():
            # tagsCollection[key]['tipo'] = 'inicial'
            for codigo, orientacion, coordenadas_tag in code_location:
                if str(int(codigo, 2)) + orientacion == key and tagsCollection[key]['tipo'] == 'inicial':
                    tagsCollection[key]['code'] = codigo
                    tagsCollection[key]['location'] = orientacion
                    tagsCollection[key]['corners'] = coordenadas_tag
                    tagsCollection[key]['tipo'] = 'actual'
                    # print(type(coordenadas_tag))

    return tagsCollection


def find_tags_subpix(frame, tagsCollection):
    dim = 200
    p1 = np.array([
        [0, 0],
        [dim - 1, 0],
        [dim - 1, dim - 1],
        [0, dim - 1]], dtype="float32")

    collection_work = tagsCollection
    for key, value in collection_work.items():
        tagsCollection[key]['tipo'] = 'inicial'

    frame_work = frame.copy()
    for imagen in range(10):
        final_contour_list = contour_generator(frame_work)
        code_location = list()

        for i in range(len(final_contour_list)):

            c_rez = final_contour_list[i][:, 0]
            H_matrix = homo(p1, order(c_rez))
            tag = cv2.warpPerspective(frame_work, H_matrix, (200, 200))

            tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
            decoded, location = get_code_location(tag1)

            if decoded is not None:
                code_location.append((decoded, location, order(final_contour_list[i][:, 0])))

        for key, value in collection_work.items():
            # tagsCollection[key]['tipo'] = 'inicial'
            for codigo, orientacion, coordenadas_tag in code_location:
                if str(int(codigo, 2)) + orientacion == key and tagsCollection[key]['tipo'] == 'inicial':
                    tagsCollection[key]['code'] = codigo
                    tagsCollection[key]['location'] = orientacion
                    tagsCollection[key]['corners'] = coordenadas_tag
                    tagsCollection[key]['tipo'] = 'actual'
                    # print(type(coordenadas_tag))

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    puntos_izq = []
    for key, value in collection_work.items():

        if tagsCollection[key]['tipo'] == 'actual':
            cornersIzq = tagsCollection[key]['corners']

            puntos_izq.append((cornersIzq[0]))
            puntos_izq.append((cornersIzq[1]))
            puntos_izq.append((cornersIzq[2]))
            puntos_izq.append((cornersIzq[3]))

            corners = np.array([puntos_izq])
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
            try:
                cv2.cornerSubPix(gray_image, corners, (3, 3), (-1, -1), criteria)
                # print(corners)
            except Exception as e:
                print("problemas subpix", e)

            tagsCollection[key]['corners'] = corners.flatten().tolist()
            puntos_izq.clear()

    return tagsCollection


def find_tags_patron_12_1(frame, tagsCollection, configuracion):
    dim = 200
    p1 = np.array([
        [0, 0],
        [dim - 1, 0],
        [dim - 1, dim - 1],
        [0, dim - 1]], dtype="float32")

    frame_work = frame.copy()

    final_contour_list = contour_generator(frame_work)
    code_location = list()

    collection_work = tagsCollection

    for i in range(len(final_contour_list)):

        c_rez = final_contour_list[i][:, 0]
        H_matrix = homo(p1, order(c_rez))
        tag = cv2.warpPerspective(frame_work, H_matrix, (200, 200))

        tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        decoded, location = get_code_location(tag1)

        if decoded is not None:
            code_location.append((decoded, location, order(final_contour_list[i][:, 0])))

    for key, value in collection_work.items():
        for codigo, orientacion, coordenadas_tag in code_location:
            if int(codigo, 2) == key:

                tagsCollection[key]['code'] = codigo
                tagsCollection[key]['location'] = orientacion
                tagsCollection[key]['corners'] = list(np.float32(np.ndarray.flatten(coordenadas_tag)))
                tagsCollection[key]['tipo'] = 'actual'

    return tagsCollection


def find_tags_patron_15_1(frame, tagsCollection, configuracion):

    dim = 200
    p1 = np.array([
        [0, 0],
        [dim - 1, 0],
        [dim - 1, dim - 1],
        [0, dim - 1]], dtype="float32")

    frame_work = frame.copy()

    # Se buscan los tags TL (horizontales) de la base Z = 0
    bandera = True
    while bandera:
        bandera = False

        final_contour_list = contour_generator(frame_work)
        code_location = list()

        collection_work = tagsCollection

        for i in range(len(final_contour_list)):

            c_rez = final_contour_list[i][:, 0]
            H_matrix = homo(p1, order(c_rez))
            tag = cv2.warpPerspective(frame_work, H_matrix, (200, 200))

            tag1 = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
            decoded, location = get_code_location(tag1)

            if decoded is not None:
                code_location.append((decoded, location, order(final_contour_list[i][:, 0])))

        for key, value in collection_work.items():
            for codigo, orientacion, coordenadas_tag in code_location:
                if str(int(codigo, 2))+'TL' == key and orientacion == 'TL':
                    tagsCollection[key]['code'] = codigo
                    tagsCollection[key]['location'] = orientacion
                    tagsCollection[key]['corners'] = list(np.float32(np.ndarray.flatten(coordenadas_tag)))
                    tagsCollection[key]['tipo'] = 'actual'

        for key, value in tagsCollection.items():
            if value['location'] == 'TL':
                if value['tipo'] != 'actual':
                    bandera = True

    # Se redimensiona la imagen de la base con tags TL a 1400 X 1400

    puntos_frame = []
    puntos_patron = []

    puntos_frame.append(((tagsCollection['1TL']['corners'][(0)]), tagsCollection['1TL']['corners'][(1)]))
    puntos_frame.append(((tagsCollection['1TL']['corners'][(4)]), tagsCollection['1TL']['corners'][(5)]))

    puntos_frame.append(((tagsCollection['9TL']['corners'][(2)]), tagsCollection['9TL']['corners'][(3)]))
    #puntos_frame.append(((tagsCollection['9TL']['corners'][(6)]), tagsCollection['9TL']['corners'][(7)]))

    puntos_frame.append(((tagsCollection['12TL']['corners'][(4)]), tagsCollection['12TL']['corners'][(5)]))
    puntos_frame.append(((tagsCollection['12TL']['corners'][(0)]), tagsCollection['12TL']['corners'][(1)]))

    puntos_frame.append(((tagsCollection['4TL']['corners'][(6)]), tagsCollection['4TL']['corners'][(7)]))
    #puntos_frame.append(((tagsCollection['4TL']['corners'][(2)]), tagsCollection['4TL']['corners'][(3)]))

    puntos_patron.append(((0), 0))
    puntos_patron.append(((200), 200))

    puntos_patron.append(((1400), 0))
    #puntos_patron.append(((1200), 200))

    puntos_patron.append(((1400), 1400))
    puntos_patron.append(((1200), 1200))

    puntos_patron.append(((0), 1400))
    #puntos_patron.append(((200), 1200))

    (H, status) = cv2.findHomography(np.array(puntos_frame), np.array(puntos_patron), 0, 5)

    for key, value in tagsCollection.items():
        if value['location'] == 'TL':

            esquinas = np.float32([[value['corners'][(0)], value['corners'][(1)]],
                                    [value['corners'][(2)], value['corners'][(3)]],
                                    [value['corners'][(4)], value['corners'][(5)]],
                                    [value['corners'][(6)], value['corners'][(7)]]]).reshape(-1, 1, 2)

            esquinas_transformadas = cv2.perspectiveTransform(esquinas, H)

            tagsCollection[key]['corners'] = (esquinas_transformadas[0][0][0], esquinas_transformadas[0][0][1],
                                                  esquinas_transformadas[1][0][0], esquinas_transformadas[1][0][1],
                                                  esquinas_transformadas[2][0][0], esquinas_transformadas[2][0][1],
                                                  esquinas_transformadas[3][0][0], esquinas_transformadas[3][0][1])

    return tagsCollection


def set_tags_patron_15_1_manual(tagsCollection):

    lista_puntos = []
    puntos_3d = []

    puntos_3d.clear()
    lista_puntos.clear()

    lista_puntos.append((0, 0, 200, 0, 200, 200, 0, 200))
    lista_puntos.append((0, 400, 200, 400, 200, 600, 0, 600))
    lista_puntos.append((0, 800, 200, 800, 200, 1000, 0, 1000))
    lista_puntos.append((0, 1200, 200, 1200, 200, 1400, 0, 1400))

    lista_puntos.append((400, 0, 600, 0, 600, 200, 400, 200))
    lista_puntos.append((400, 1200, 600, 1200, 600, 1400, 400, 1400))

    lista_puntos.append((800, 0, 1000, 0, 1000, 200, 800, 200))
    lista_puntos.append((800, 1200, 1000, 1200, 1000, 1400, 800, 1400))

    lista_puntos.append((1200, 0, 1400, 0, 1400, 200, 1200, 200))
    lista_puntos.append((1200, 400, 1400, 400, 1400, 600, 1200, 600))
    lista_puntos.append((1200, 800, 1400, 800, 1400, 1000, 1200, 1000))
    lista_puntos.append((1200, 1200, 1400, 1200, 1400, 1400, 1200, 1400))

    for i in range(12):
        puntos_3d.clear()

        tagsCollection[str(i+1)+'TL']['corners'] = lista_puntos[i]

        puntos_3d.append((lista_puntos[i][0], lista_puntos[i][(1)], 0))
        puntos_3d.append((lista_puntos[i][2], lista_puntos[i][(3)], 0))
        puntos_3d.append((lista_puntos[i][4], lista_puntos[i][(5)], 0))
        puntos_3d.append((lista_puntos[i][6], lista_puntos[i][(7)], 0))

        tagsCollection[str(i+1) + 'TL']['corners_3D'] = list(puntos_3d)

        tagsCollection[str(i + 1) + 'TL']['tipo'] = 'estimado'

    lista_puntos.clear()
    puntos_3d.clear()

    lista_puntos.append((0, 0, 200, 0, 200, 200, 0, 200))
    tagsCollection['1BL']['corners'] = lista_puntos[0]
    puntos_3d.append((0, 1410, -10))
    puntos_3d.append((200, 1410, -10))
    puntos_3d.append((200, 1410, -210))
    puntos_3d.append((0, 1410, -210))
    tagsCollection['1BL']['corners_3D'] = list(puntos_3d)
    tagsCollection['1BL']['tipo'] = 'estimado'

    lista_puntos.clear()
    lista_puntos.append((600, 0, 800, 0, 800, 200, 600, 200))
    tagsCollection['3BL']['corners'] = lista_puntos[0]
    puntos_3d.clear()
    puntos_3d.append((600, 1410, -10))
    puntos_3d.append((800, 1410, -10))
    puntos_3d.append((800, 1410, -210))
    puntos_3d.append((600, 1410, -210))
    tagsCollection['3BL']['corners_3D'] = list(puntos_3d)
    tagsCollection['3BL']['tipo'] = 'estimado'

    lista_puntos.clear()
    lista_puntos.append((1200, 0, 1400, 0, 1400, 200, 1200, 200))
    tagsCollection['5BL']['corners'] = lista_puntos[0]
    puntos_3d.clear()
    puntos_3d.append((1200, 1410, -10))
    puntos_3d.append((1400, 1410, -10))
    puntos_3d.append((1400, 1410, -210))
    puntos_3d.append((1200, 1410, -210))
    tagsCollection['5BL']['corners_3D'] = list(puntos_3d)
    tagsCollection['5BL']['tipo'] = 'estimado'

    return tagsCollection


def set_tags_patron_16_1_manual(tagsCollection):

    lista_puntos = []
    puntos_3d = []

    puntos_3d.clear()
    lista_puntos.clear()

    lista_puntos.append((0, 0, 2000, 0, 2000, -2000, 0, -2000))
    lista_puntos.append((0, -4000, 2000, -4000, 2000, -6000, 0, -6000))
    lista_puntos.append((0, -8000, 2000, -8000, 2000, -10000, 0, -10000))
    lista_puntos.append((0, -12000, 2000, -12000, 2000, -14000, 0, -14000))

    lista_puntos.append((4000, 0, 6000, 0, 6000, -2000, 4000, -2000))
    lista_puntos.append((4000, -12000, 6000, -12000, 6000, -14000, 4000, -14000))

    lista_puntos.append((8000, 0, 10000, 0, 10000, -2000, 8000, -2000))
    lista_puntos.append((8000, -12000, 10000, -12000, 10000, -14000, 8000, -14000))

    lista_puntos.append((12000, 0, 14000, 0, 14000, -2000, 12000, -2000))
    lista_puntos.append((12000, -4000, 14000, -4000, 14000, -6000, 12000, -6000))
    lista_puntos.append((12000, -8000, 14000, -8000, 14000, -10000, 12000, -10000))
    lista_puntos.append((12000, -12000, 14000, -12000, 14000, -14000, 12000, -14000))

    for i in range(12):
        puntos_3d.clear()

        tagsCollection[str(i+1)+'TL']['corners'] = lista_puntos[i]

        puntos_3d.append((lista_puntos[i][0], lista_puntos[i][(1)], 0))
        puntos_3d.append((lista_puntos[i][2], lista_puntos[i][(3)], 0))
        puntos_3d.append((lista_puntos[i][4], lista_puntos[i][(5)], 0))
        puntos_3d.append((lista_puntos[i][6], lista_puntos[i][(7)], 0))

        tagsCollection[str(i+1) + 'TL']['corners_3D'] = list(puntos_3d)

        tagsCollection[str(i + 1) + 'TL']['tipo'] = 'estimado'

    lista_puntos.clear()
    puntos_3d.clear()

    lista_puntos.append((0, 0, 2000, 0, 2000, 2000, 0, 2000))
    tagsCollection['1BL']['corners'] = lista_puntos[0]
    puntos_3d.append((0, -14150, -150))
    puntos_3d.append((2000, -14150, -150))
    puntos_3d.append((2000, -14150, -2150))
    puntos_3d.append((0, -14150, -2150))
    tagsCollection['1BL']['corners_3D'] = list(puntos_3d)
    tagsCollection['1BL']['tipo'] = 'estimado'

    lista_puntos.clear()
    lista_puntos.append((4000, 0, 6000, 0, 6000, 2000, 4000, 2000))
    tagsCollection['2BL']['corners'] = lista_puntos[0]
    puntos_3d.clear()
    puntos_3d.append((4000, -14150, -150))
    puntos_3d.append((6000, -14150, -150))
    puntos_3d.append((6000, -14150, -2150))
    puntos_3d.append((4000, -14150, -2150))
    tagsCollection['2BL']['corners_3D'] = list(puntos_3d)
    tagsCollection['2BL']['tipo'] = 'estimado'

    lista_puntos.clear()
    lista_puntos.append((8000, 0, 10000, 0, 10000, 2000, 8000, 2000))
    tagsCollection['3BL']['corners'] = lista_puntos[0]
    puntos_3d.clear()
    puntos_3d.append((8000, -14150, -150))
    puntos_3d.append((10000, -14150, -150))
    puntos_3d.append((10000, -14150, -2150))
    puntos_3d.append((8000, -14150, -2150))
    tagsCollection['3BL']['corners_3D'] = list(puntos_3d)
    tagsCollection['3BL']['tipo'] = 'estimado'

    lista_puntos.clear()
    lista_puntos.append((12000, 0, 14000, 0, 14000, 2000, 12000, 2000))
    tagsCollection['4BL']['corners'] = lista_puntos[0]
    puntos_3d.clear()
    puntos_3d.append((12000, -14150, -150))
    puntos_3d.append((14000, -14150, -150))
    puntos_3d.append((14000, -14150, -2150))
    puntos_3d.append((12000, -14150, -2150))
    tagsCollection['4BL']['corners_3D'] = list(puntos_3d)
    tagsCollection['4BL']['tipo'] = 'estimado'

    return tagsCollection


def inicializar_tags_12_1(nTags = 12):
    collection_tags = {}
    for n in range(nTags):
        kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos(n_puntos = 4)
        collection_tags[n+1] = {'code': None, 'location': None, 'corners': None, 'kalman': kalman, 'measurement_array': measurement_array, 'dt_array': dt_array, 'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

    kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos(n_puntos=4)
    collection_tags[15] = {'code': None, 'location': None, 'corners': None, 'kalman': kalman, 'measurement_array': measurement_array, 'dt_array': dt_array, 'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

    return collection_tags


def inicializar_tags_15_1(nTagsBaseTL = 12, nTagsBaseBL=3, nTagsPunta = 1):
    collection_tags = {}
    for n in range(nTagsBaseTL):
        kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos(n_puntos = 4)
        collection_tags[str(n+1)+'TL'] = {'code': None, 'location': 'TL', 'corners': None, 'corners_3D': None,
                                          'kalman': kalman, 'measurement_array': measurement_array,
                                          'dt_array': dt_array, 'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

    for n in range(nTagsBaseBL):
        kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos(n_puntos = 4)
        collection_tags[str((n*2)+1)+'BL'] = {'code': None, 'location': 'BL', 'corners': None, 'corners_3D': None,
                                          'kalman': kalman, 'measurement_array': measurement_array,
                                          'dt_array': dt_array, 'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

    kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos(n_puntos=4)
    collection_tags[str(15)+'BL'] = {'code': None, 'location': 'BL', 'corners': None, 'corners_3D': None, 'kalman': kalman,
                           'measurement_array': measurement_array, 'dt_array': dt_array,
                           'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

    return collection_tags


def inicializar_tags_16_1(nTagsBaseTL = 12, nTagsBaseBL=4, nTagsPunta = 1):
    collection_tags = {}
    for n in range(nTagsBaseTL):
        kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos(n_puntos = 4)
        collection_tags[str(n+1)+'TL'] = {'code': None, 'location': 'TL', 'corners': None, 'corners_3D': None,
                                          'kalman': kalman, 'measurement_array': measurement_array,
                                          'dt_array': dt_array, 'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

    for n in range(nTagsBaseBL):
        kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos(n_puntos = 4)
        collection_tags[str(n+1)+'BL'] = {'code': None, 'location': 'BL', 'corners': None, 'corners_3D': None,
                                          'kalman': kalman, 'measurement_array': measurement_array,
                                          'dt_array': dt_array, 'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

    kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos(n_puntos=4)
    collection_tags[str(15)+'BL'] = {'code': None, 'location': 'BL', 'corners': None, 'corners_3D': None, 'kalman': kalman,
                           'measurement_array': measurement_array, 'dt_array': dt_array,
                           'tiempo_estimacion': time.time(), 'tipo': 'inicial'}

    return collection_tags


def inicializar_coleccion_punta():
    colecion_punta = {}

    kalman, measurement_array, dt_array = gestionKalman.inicializar_kalman_multiples_puntos_3D(n_puntos=1)

    colecion_punta['x'] = 0
    colecion_punta['y'] = 0
    colecion_punta['z'] = 0
    colecion_punta['xyz'] = []
    colecion_punta['kalman'] = kalman
    colecion_punta['measurement_array'] = measurement_array
    colecion_punta['tiempo_estimacion'] = time.time()
    colecion_punta['dt_array'] = dt_array
    colecion_punta['P1'] = None
    colecion_punta['P2'] = None


def inicializar_tags(configuracion=(16, 1)):
    collection_tags = {}

    if configuracion == (12, 1):
        collection_tags = inicializar_tags_12_1(12)

    if configuracion == (15, 1):
        collection_tags = inicializar_tags_15_1(12, 3, 1)

    if configuracion == (16, 1):
        collection_tags = inicializar_tags_16_1(12, 4, 1)

    return collection_tags


def find_tags_patron(frame, tagsCollection, configuracion):
    if configuracion == (12, 1):
        find_tags_patron_12_1(frame, tagsCollection, configuracion)
    if configuracion == (15, 1):
        find_tags_patron_15_1(frame, tagsCollection, configuracion)


def visualizar_tags():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    # cap1 = cv2.VideoCapture('izq.avi')
    # cap2 = cv2.VideoCapture('der.avi')

    dim = 200
    p1 = np.array([
        [0, 0],
        [dim - 1, 0],
        [dim - 1, dim - 1],
        [0, dim - 1]], dtype="float32")

    frame3 = cv2.imread('patronBase.jpg')

    while True:
        # Capture frame-by-frame
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        frame3 = cv2.imread('patronBase.jpg')
        test_img1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        test_blur = cv2.GaussianBlur(test_img1, (3, 3), 0)
        edge = cv2.Canny(test_blur, 60, 180)
        # tags = inicializar_tags_15_1(12, 3, 1)
        # tags = set_tags_patron_15_1_manual(tags)
        # print(tags['1TL']['corners_3D'])
        # print(tags['1BL']['corners_3D'])

        try:
            # coleccionTags = get_listado_tags(frame1)
            frame1 = image_process(frame1, p1)
            frame2 = image_process(frame2, p1)
            frame3 = image_process(frame3, p1)
        except:
            print("Error ")

        # Display the resulting frame
        cv2.imshow('frame 1', frame1)
        cv2.imshow('frame 2', frame2)
        cv2.imshow('frame 3', frame3)
        cv2.imshow('Gray', edge)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


def main():
    visualizar_tags()


if __name__ == '__main__':
    main()