# -*- coding: utf-8 -*-
import cv2, os, copy
import numpy as np
from numpy import linalg as LA
from scipy.stats import norm
from matplotlib import pyplot as plt
import time

class ScanBook(object):
	class Book(object):
		class Region(object):
			def __init__(self, book, original, reference, matrix, size):
				self.__parent = book
				self.original = original
				self.reference = reference
				self.matrix = matrix
				_, self.invmatrix = cv2.invert(matrix)
				self.size = size

			def crop(self):
				bb = self.__parent.rect2bb(self.reference)
				return self.__parent.imReference[bb[2]:bb[3], bb[0]:bb[1], :]
				


		class Spine(object):
			def __init__(self, book, detected, original, reference):
				self.__parent = book
				self.detected = detected
				self.original = original
				self.reference = reference


		class Page(object):
			def __init__(self, book, reference):
				self.__parent = book
				self.reference = reference
				self.original = ScanBook.applyHomoInPoints(self.reference, self.__parent.region.invmatrix)

			def crop(self):
				bb = self.__parent.rect2bb(self.reference)
				return self.__parent.imReference[bb[2]:bb[3], bb[0]:bb[1], :]

				

		def __init__(self, im, name, typeOfScan, approxLinesOfText,
			regionOriginal, regionReference, regionMatrix, regionSize,
			spineDetected, spineOriginal, spineReference,
			pagesListOfPages):
			self.im = im
			self.name = name
			self.typeOfScan = typeOfScan
			self.approxLinesOfText = approxLinesOfText

			self.region = self.Region(self, regionOriginal, regionReference, regionMatrix, regionSize)
			self.spine = self.Spine(self, spineDetected, spineOriginal, spineReference)
			self.pages = [self.Page(self, p) for p in pagesListOfPages]

			self.imReference = cv2.warpPerspective(self.im, regionMatrix, regionSize)

		def save(self, name, page = None):
			nn = name.split(".")
			if page is None:
				i = 0
				for p in self.pages:
					cv2.imwrite(nn[0]+'_'+str(i)+'.'+nn[1], p.crop())
					i += 1
			else:
				cv2.imwrite(name, self.pages[i].crop())

		def summary(self, name):
			N = len(self.pages)

			imout = self.im.copy()
			
			# each region
			i = 1
			for p in self.pages:
				out = p.crop()
				cv2.drawContours(imout, [np.int32(p.original)], -1, (0, 255 , 0), 15)
				plt.subplot(int("1" + str(N + 1) + str(i + 1)))
				plt.imshow(cv2.resize(out, (int(0.4*out.shape[1]), int(0.4*out.shape[0]))),'gray'), plt.title('Capture num'+str(i))
				i += 1

			# if has spine draw
			if self.spine.detected:
				cv2.line(imout, 
						 (int(self.spine.original[0, 0]), int(self.spine.original[0, 1])), 
						 (int(self.spine.original[1, 0]), int(self.spine.original[1, 1])), 
						 (255, 0 , 0), 
						 15)
				


			# real image with regions
			plt.subplot(int("1"+ str(N + 1) + "1"))
			plt.imshow(cv2.resize(imout, (int(0.4*imout.shape[1]), int(0.4*imout.shape[0]))),'gray'), 
			plt.title('Original [hasSpine='+str(self.spine.detected)+",\n pageOrientation="+('horizontal' if self.typeOfScan else 'vertical')+"]")
			plt.show(block = False)
			plt.savefig(name+'.pdf') 

		def rect2bb(self, r):
			return (int(r[0][0]), int(r[1][0]), int(r[0][1]), int(r[2][1]))

					
				

	def __init__(self, params):
		self.__params = params

	def scan(self, filename):
		if not os.path.exists(filename):
			raise IOError('File not exist')
		im = cv2.imread(filename)
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		
		# obtenemos la region y los parametros asociados
		rect, rectReference, matrix, sizeImage, typeOfScan, numberOfLines = self.__getRegion(gray)
		_, invmatrix = cv2.invert(matrix)
		"""book = {'region': 
					{'original': np.reshape(rect, [rect.shape[0], 2]),
					 'reference': np.reshape(rectReference, [rectReference.shape[0], 2]),
					 'size': sizeImage,
					 'typeOfScan': typeOfScan,
					 'aproximationLinesOfText': numberOfLines}
				}"""

		# determinamos si tiene spine o no
		grayPerspective = cv2.warpPerspective(gray, matrix, sizeImage)
		hasSpine, location = self.__spineLocation(grayPerspective)
		if hasSpine:
			rectReferenceList = self.__splitPage(rect, rectReference, location)
			locPoints = np.matrix([[location, 0], [location, sizeImage[1]]])
			realLocPoints = ScanBook.applyHomoInPoints(locPoints, invmatrix)
			spineobj = {'detected': hasSpine, 'original': realLocPoints, 'reference': locPoints}
		else:
			rectReferenceList = [np.reshape(rectReference, [rectReference.shape[0], 2])]
			spineobj = {'detected': False, 'original': [], 'reference': []}

		book = \
		self.Book(im,
				  filename,
				  typeOfScan,
				  numberOfLines,
				  # region
				  np.reshape(rect, [rect.shape[0], 2]), 
				  np.reshape(rectReference, [rectReference.shape[0], 2]),
				  matrix,
				  sizeImage,
				  #spine
				  spineobj['detected'], 
				  spineobj['original'], 
				  spineobj['reference'],
				  #pages
				  rectReferenceList)
		return book

	def __homopoint(self, p):
		aux = p[0], p[1], 1
		return aux

	@staticmethod
	def applyHomoInPoints(pointsIn, mat):
		pointsIn = copy.deepcopy(pointsIn)
		points = np.hstack([pointsIn, np.ones(shape=(pointsIn.shape[0], 1))])
		pointsH = np.matmul(mat, points.T)
		pointsH = pointsH.T
		z_scale = np.tile(pointsH[:, 2].T, [3, 1]).T
		pointsH /= z_scale
		return pointsH[:, :2]

	def __gray2rgb(self, gray):
		gray = np.uint8(np.tile(np.reshape(gray, [gray.shape[0], gray.shape[1], 1]), [1, 1, 3]))
		return gray

	# detect region of the book
	def __getRegion(self, gray):
		def getRectangle(otsu):
			def prefilter(otsu):
				# prefiltrado
				otsucpy = copy.deepcopy(otsu)
				# fill holes
				contour, hier = cv2.findContours(otsucpy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
				for cnt in contour:
				    cv2.drawContours(otsucpy, [cnt], 0, 255, -1)

				# remove again noise
				open_size = self.__params['detect_region']['remove_noise_open']
				otsucpy = cv2.morphologyEx(otsucpy, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
				return otsucpy

			# consts
			coef_angle = (np.pi*self.__params['detect_region']['ortogonal_error_coef_in_deg'])/180

			# prefilter
			bw = prefilter(otsu)

			# obtenemos los contornos
			contour, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
			maxContour = None
			maxArea = 0
			for cnt in contour:
				area = cv2.contourArea(cnt)
				if area > maxArea:
					maxArea = area
					maxContour = cnt

			# una vez se tiene la area mas grande se procede a obtener los puntos interesantes
			points = []

			# sino hay contorno maximo, no se puede hacer nada
			if maxContour is None:
				raise IOError('Region not detected')
			
			# Detect the points of the polygon
			hull = cv2.convexHull(maxContour, returnPoints= False)
			pointsHull = maxContour[hull[:, 0]]
			pointsRaw = cv2.approxPolyDP(pointsHull, self.__params['detect_region']['approx_poly_arc_weight'] * cv2.arcLength(pointsHull, True), True)

			# parseamos los puntos que
			#	- generen lineas que tengan un angulo de 90 grados
			#	- solo queremos 4 puntos
			auxLines = []
			for i in xrange(len(pointsRaw)):
				j = (i + 1) % len(pointsRaw)
				h1 = np.float32(self.__homopoint(pointsRaw[i][0]))
				h2 = np.float32(self.__homopoint(pointsRaw[j][0]))
				line = np.cross(h1, h2)
				line = line/line[2]
				auxLines.append(line)

			# Calculamos las lineas que cumplen la restriccion
			for i in xrange(len(auxLines)):
				if len(points) >= 4:
					break
				j = (i + 1) % len(auxLines)
				l1 = auxLines[i][:2]
				l2 = auxLines[j][:2]
				l1_length = LA.norm(l1)
				l2_length = LA.norm(l2)

				# Calculamos el angulo entre rectas
				angle = np.arccos(np.dot(l1, l2)/(l1_length*l2_length))
				if np.abs(angle - (np.pi/2)) < coef_angle:
					# Si hay interseccion calculamos 
					#p = np.cross(auxLines[i], auxLines[j])
					#p /= p[2]
					# es equivalente a, asi no es necesario calcularlo
					p = pointsRaw[j][0]
					points.append(p[:2])
			# devolvemos los puntos
			points = np.array(points).reshape((-1,1,2)).astype(np.float32)
			points = np.roll(points, -2)

			return points

		def sortRectangleRegion(otsu, rect):
			def determineOrientationRectangle(rect):
				verticalLength = int(LA.norm(rect[3][0] - rect[0][0]))
				horizontalLength = int(LA.norm(rect[0][0] - rect[1][0]))
				rectReference = np.array([(0, 0), (horizontalLength, 0), (horizontalLength, verticalLength),(0, verticalLength)], np.float32)
				matrix = cv2.getPerspectiveTransform(rect, rectReference)
				size = (horizontalLength, verticalLength)
				return rectReference, matrix, size

			def determineOrientationText(analizeText):
				# True Horizontal
				# False Vertical
				# aplicamos un open invertido (top hat), para eliminar pequeños elementos
				remove_noise_inv_open = self.__params['detect_region']['detect_text']['remove_noise_inv_open']
				remove_noise_open = self.__params['detect_region']['detect_text']['remove_noise_open']
				bw = 255 - np.float32(cv2.morphologyEx(analizeText, cv2.MORPH_OPEN, np.ones((remove_noise_inv_open, remove_noise_inv_open), np.uint8)))
				
				# aplicamos un open
				bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((remove_noise_open, remove_noise_open), np.uint8))
				
				# obtemos los edges
				edges = cv2.Canny(np.uint8(bw), 50, 10, apertureSize= 3)

				# obtenemos las lineas mediante hough
				lines = cv2.HoughLinesP(image= edges, rho= 1, theta= np.pi/180, threshold= self.__params['detect_region']['resize_height']/10, lines= np.array([]), minLineLength= self.__params['detect_region']['resize_height']/5, maxLineGap= self.__params['detect_region']['resize_height']/6)
				
				accumHorizontal = 0
				accumVertical = 0
				
				for l in lines[0]:
					p1 = l[0:2]
					p2 = l[2:4]
					r = np.abs(p2 - p1)

					# determinamos si la linea entre los puntos es horizontal o vertical
					if r[0] > r[1]:
						accumHorizontal += 1
					else:
						accumVertical += 1
				if accumHorizontal > accumVertical:
					return True, accumHorizontal
				else:
					return False, accumVertical

			# Calculamos la matriz de transformacion
			rectReference, matrix, size = determineOrientationRectangle(rect)
			analizeText = cv2.warpPerspective(otsu, matrix, size)
			#		  B
			#      |------|
			#   -  x------x 1
			#   \  \ 0
			# A \  \
			#   \  \
			#   -  x 3      2
			# A = vertical length
			# B = horizontal length
			oriText, numberOfLines = determineOrientationText(analizeText)
			if not oriText:
				# Rotamos el contenedor
				rect = np.roll(rect, -2)
				rectReference, matrix, size = determineOrientationRectangle(rect)

			if size[0] > size[1]:
				typeOfScan = True
			else:
				typeOfScan = False

			# devolvemos los valores
			return rect, rectReference, matrix, size, typeOfScan, numberOfLines

		def reScalePoints(points, wS, hS):
			return points*np.tile(np.array([[(wS, hS)]], np.float32), [rect.shape[0], 1, 1])

		# params
		resize_h = self.__params['detect_region']['resize_height']

		# primero reescalamos la imagen
		hOrig, wOrig = gray.shape
		hNew, wNew = resize_h, int(wOrig*resize_h/hOrig)
		gray = cv2.resize(gray, (wNew, hNew))
		_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		# get rectangle region
		rect = getRectangle(otsu)
		
		# la posicion de la region no tiene porque se correcta, debemos reordenarla
		rect, rectReference, matrix, sizeImage, typeOfScan, numberOfLines = sortRectangleRegion(otsu, rect)
		
		# reescalmos todos los valores	
		wS =  wOrig/float(wNew)
		hS = hOrig/float(hNew)
		rect = reScalePoints(rect, wS, hS)
		rectReference = reScalePoints(np.reshape(rectReference, [rectReference.shape[0],1,2]), wS, hS)
		matrix = cv2.getPerspectiveTransform(rect, rectReference)
		sizeImage = (int(hOrig*float(sizeImage[0])/hNew), int(wOrig*float(sizeImage[1])/wNew))
		return rect, rectReference, matrix, sizeImage, typeOfScan, numberOfLines

	# detect if has spine the book
	def __spineLocation(self, grayPerspective):
		def normPdfPositionSpine(x, width):
			mu = width/2
			sigma = width*0.3
			return norm.pdf(x, mu, sigma)

		resize_h = self.__params['detect_spine']['resize_height']
		hOrig, wOrig = grayPerspective.shape
		hNew, wNew = resize_h, int(wOrig*resize_h/hOrig)
		grayPerspective = cv2.resize(grayPerspective, (wNew, hNew))

		searchSpine = 255 - grayPerspective
		
		# search de text to substract
		params_cascade = self.__params['detect_spine']['params_cascade']
		asolateText = cv2.morphologyEx(searchSpine, cv2.MORPH_DILATE, np.ones((params_cascade[0], params_cascade[0]), np.uint8))
		_, asolateText = cv2.threshold(asolateText, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		asolateText = cv2.morphologyEx(asolateText, cv2.MORPH_OPEN, np.ones((1, params_cascade[1]), np.uint8))
		asolateText = cv2.morphologyEx(asolateText, cv2.MORPH_DILATE, np.ones((params_cascade[2], params_cascade[2]), np.uint8))

		# asolate the spine
		asolateSpine = cv2.morphologyEx(searchSpine, cv2.MORPH_OPEN, np.ones((params_cascade[3], 1), np.uint8))
		asolateSpine = cv2.morphologyEx(asolateSpine, cv2.MORPH_CLOSE, np.ones((params_cascade[4], 1), np.uint8))
		ret, asolateSpine = cv2.threshold(asolateSpine, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		asolateSpine[asolateText > 0] = 0

		#asolateSpine = cv2.Canny(asolateSpine, 50, 10, apertureSize=3)
 
		# search de possible lines
		lines = cv2.HoughLinesP(image= asolateSpine, rho= 1, theta= np.pi/180, threshold= 200, lines= np.array([]), minLineLength= 0.1*searchSpine.shape[0], maxLineGap= 100)
		location = None
		alpha = np.float( self.__params['detect_spine']['range_percent'])
		minLimitX = alpha* wNew
		maxLimitX = (1-alpha)* wNew
		
		if lines is None:
			return False, 0

		ddd = copy.deepcopy(asolateSpine)
		ddd = self.__gray2rgb(ddd)
		for l in lines[0]:
			p1 = l[0:2]
			p2 = l[2:4]
			r = np.abs(p2 - p1)
			pmean = np.mean([p1, p2], axis=0)
			angle = 180*np.arctan2(r[1], r[0])/np.pi

			if r[1] > r[0] and \
			np.abs(angle - 90) < 15 and \
			pmean[0] > minLimitX and \
			pmean[0] < maxLimitX:
				if location is None:
					location = np.zeros(shape=(0, 2))
				weight = normPdfPositionSpine(pmean[0], wNew)*LA.norm(r)
				location = np.vstack([location, np.array([pmean[0], weight])])

		if location is not None:
			sumL = np.sum(location[:, 1])
			if sumL > 0:
				locationMean = np.sum(location[:, 0]*location[:, 1])/sumL
				hasSpine = True
			else:
				locationMean = 0
				hasSpine = False
		else:
			locationMean = 0
			hasSpine = False

		return hasSpine, wOrig*locationMean/wNew

	def __splitPage(self, rect, rectRef, location):
		rect = np.reshape(rect, [rect.shape[0], 2])
		rectRef = np.reshape(rectRef, [rectRef.shape[0], 2])

		rectA = copy.deepcopy(rectRef)
		rectA[0] = rectRef[0][0]
		rectA[1][:] = location, rectRef[1][1]
		rectA[2][:] = location, rectRef[2][1]
		rectA[3] = rectRef[3]
		
		rectB = copy.deepcopy(rectRef)
		rectB[0][:] = location, rectRef[0][1]
		rectB[1] = rectRef[1]
		rectB[2] = rectRef[2]
		rectB[3][:] = location, rectRef[3][1]
		
		return rectA, rectB

params = {'detect_region': 
				{'resize_height': 500,
				 'remove_noise_open': 5,
				 'approx_poly_arc_weight': 0.04,
				 'ortogonal_error_coef_in_deg': 10,
				 'detect_text':
		  			{'remove_noise_inv_open': 3,
		  		 	'remove_noise_open': 3}
		  		 },
		  'detect_spine':
		  		{'resize_height': 500,
		  		 'params_cascade': [11, 25, 25, 20, 15],
		  		 'range_percent': 0.2}
		 }
s = ScanBook(params)
a1 = time.time()
for n in os.listdir('images'):
	if cv2.imread('images/'+n) is None:
		continue

	a0 = time.time()
	book = s.scan('images/'+n)
	book.save('results/'+n)
	book.summary('summarys/'+n.split(".")[0])
	print time.time() - a0
print 'Total:',time.time() - a1