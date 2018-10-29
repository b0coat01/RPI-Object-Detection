"""

<Description>
This is a blob detection program which intend to find the biggest blob
in a given picture taken by a camera and return its central positon with
bounding box. It tracks one main reference object and N-objects fitting 
into a second criteria.

The citeria for detection is color-based and can be modified as needed
to fit any other application. 

Key Steps:
[1] Image Filtering
[2] Image Segmentation
[3] Detect Blobs
[4] Filter Blobs using a criteria
[5] Track Blobs
</Description>

<Author>
Brandon Coats
TG Automation
www.tgautomation.tech


"""

import cv2
import numpy as np
import collections
import glob

cart_width=1.5
cart_length=3.0
img_index = 0
img_list = glob.glob('images/*.jpg')
img_list.sort()
print('img_list', img_list)

font = cv2.FONT_HERSHEY_TRIPLEX

w, h = 3, 3;
mat_inv = [[0 for x in range(w)] for y in range(h)] 


def isset(v):
	try:
		type (eval(v))
	except:
		return 0
	else:
		return 1

# create video capture
cap = cv2.VideoCapture(0)

while(1):

	# Read the frames frome a camera
	#_,frame = cap.read()
	#frame = cv2.blur(frame,(3,3))

	# Or get it from a JPEG

	if img_index > 17: 
		img_index = 0

	frame = cv2.imread(img_list[img_index], 1)
	dst = cv2.pyrDown(frame);
	frame = cv2.pyrDown(dst);
	cv2.imshow('Cart_Output', frame);
	frame_ori = frame.copy()

	# Convert the image to hsv space and find range of colors
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# This is for YELLOW tag
	thresh_yellow = cv2.inRange(hsv,np.array((20, 80, 70)), np.array((60, 255, 255)))

	# This is for BLACK tag
	thresh_black = cv2.inRange(hsv,np.array((0, 0, 0)), np.array((100, 100, 100)))

	thresh_cart = cv2.add(thresh_yellow,thresh_black)
	cv2.imshow('thresh_cart', thresh_cart)


	# find contours in the threshold image
	im2, contours,hierarchy = cv2.findContours(thresh_cart, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# finding contour with maximum area and store it as best_cnt
	max_area = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > max_area:
			max_area = area
			best_cnt = cnt
			#print("max_area: ", max_area)

	if isset('best_cnt'):
		M = cv2.moments(best_cnt)
		cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
		cnt_center = [cx,cy]
		##print("Center of ROI: ", cnt_center)

	w, h = 2, 4;
	cart_vert_list = [[0 for x in range(w)] for y in range(h)]
	max_pos_x = 0
	max_pos_y = 0
	min_pos_x = 99999
	min_pos_y = 99999
	for cnt in best_cnt:
		new_cnt=cnt[0]
		#print("new_cnt: ", new_cnt)
		contour_x = new_cnt[0]
		contour_y = new_cnt[1]
		if  (contour_x < cnt_center[0] and contour_y < cnt_center[1]):
			if (contour_x < min_pos_x and contour_y < min_pos_y):
				min_pos_x = contour_x
				#min_pos_y = contour_y
				#print("New max_pos_x: ", max_pos_x)
				cart_vert_list[0][0] = contour_x
				cart_vert_list[0][1] = contour_y
				if (contour_x < min_pos_x):
					min_pos_x = contour_x
					min_pos_y = contour_y
					#print("New max_pos_x: ", max_pos_x)
					cart_vert_list[0][0] = contour_x
					cart_vert_list[0][1] = contour_y
	max_pos_x = 0
	max_pos_y = 0
	min_pos_x = 99999
	min_pos_y = 99999
	for cnt in best_cnt:
		new_cnt=cnt[0]
		#print("new_cnt: ", new_cnt)
		contour_x = new_cnt[0]
		contour_y = new_cnt[1]
		if  (contour_x > cnt_center[0] and contour_y > cnt_center[1]):
			if (contour_x > max_pos_x and contour_y > max_pos_y):
				max_pos_x = contour_x				
				#max_pos_y = contour_y
				#print("New min_pos_x: ", min_pos_x)
				cart_vert_list[1][0] = contour_x
				cart_vert_list[1][1] = contour_y
				if (contour_x > max_pos_x):
					max_pos_y = contour_y
					max_pos_x = contour_x
					#print("New min_pos_x: ", min_pos_x)
					cart_vert_list[1][0] = contour_x
					cart_vert_list[1][1] = contour_y
	max_pos_x = 0
	max_pos_y = 0
	min_pos_x = 99999
	min_pos_y = 99999
	for cnt in best_cnt:
		new_cnt=cnt[0]
		#print("new_cnt: ", new_cnt)
		contour_x = new_cnt[0]
		contour_y = new_cnt[1]
		if  (contour_x > cnt_center[0] and contour_y < cnt_center[1]):
			if (contour_x > max_pos_x and contour_y < min_pos_y):		
				max_pos_x = contour_x			
				#min_pos_y = contour_y
				#print("New max_pos_y: ", max_pos_y)
				cart_vert_list[2][0] = contour_x
				cart_vert_list[2][1] = contour_y
				if (contour_x > max_pos_x):
					max_pos_x  = contour_x
					min_pos_y = contour_y
					#print("New min_pos_y: ", min_pos_y)
					cart_vert_list[2][0] = contour_x
					cart_vert_list[2][1] = contour_y
	max_pos_x = 0
	max_pos_y = 0
	min_pos_x = 99999
	min_pos_y = 99999
	for cnt in best_cnt:
		new_cnt=cnt[0]
		#print("new_cnt: ", new_cnt)
		contour_x = new_cnt[0]
		contour_y = new_cnt[1]
		if  (contour_x < cnt_center[0] and contour_y > cnt_center[1]):
			if (contour_x < min_pos_x and contour_y > max_pos_y):
				min_pos_x  = contour_x
				#max_pos_y = contour_y
				#print("New min_pos_y: ", min_pos_y)
				cart_vert_list[3][0] = contour_x
				cart_vert_list[3][1] = contour_y
				if (contour_x < min_pos_x):
					min_pos_x  = contour_x
					max_pos_y = contour_y
					#print("New min_pos_y: ", min_pos_y)
					cart_vert_list[3][0] = contour_x
					cart_vert_list[3][1] = contour_y

	##print("Cart Vertices: ", cart_vert_list)

	cart_vert_lr_x=cart_vert_list[0][0]
	cart_vert_lr_y=cart_vert_list[0][1]
	cart_vert_ul_x=cart_vert_list[1][0]
	cart_vert_ul_y=cart_vert_list[1][1]
	cart_vert_ll_x=cart_vert_list[2][0]
	cart_vert_ll_y=cart_vert_list[2][1]
	cart_vert_ur_x=cart_vert_list[3][0]
	cart_vert_ur_y=cart_vert_list[3][1]

	#Geometric Analysis
	len1=((cart_vert_lr_x-cart_vert_ll_x)**2+(cart_vert_lr_y-cart_vert_ll_y)**2)**0.5
	len2=((cart_vert_ur_x-cart_vert_ul_x)**2+(cart_vert_ur_y-cart_vert_ul_y)**2)**0.5
	wid1=((cart_vert_ul_x-cart_vert_ll_x)**2+(cart_vert_ul_y-cart_vert_ll_y)**2)**0.5
	wid2=((cart_vert_ur_x-cart_vert_lr_x)**2+(cart_vert_ur_y-cart_vert_lr_y)**2)**0.5
	if (len1 == 0 and len2 == 0):
		len_avg = 99999
	else:
		if (len1 == 0):
			len_avg = len2
		if (len2 == 0):
			len_avg = len1
		if (len1 != 0 and len2 != 0):
			len_avg = (len1+len2)/2
	if (wid1 == 0 and wid2 == 0):
		wid_avg = 99999
	else:
		if (wid1 == 0):
			wid_avg = wid2
		if (wid2 == 0):
			wid_avg = wid1
		if (wid1 != 0 and wid2 != 0):
			wid_avg = (wid1+wid2)/2
	##print("Cart Average Length: ", len_avg)
	##print("Cart Average Width: ", wid_avg)

	safe_dist = int(len_avg*.05)
	cart_vert_lr_x_safe=cart_vert_list[0][0]-safe_dist
	cart_vert_lr_y_safe=cart_vert_list[0][1]-safe_dist
	cart_vert_ul_x_safe=cart_vert_list[1][0]+safe_dist
	cart_vert_ul_y_safe=cart_vert_list[1][1]+safe_dist
	cart_vert_ll_x_safe=cart_vert_list[2][0]+safe_dist
	cart_vert_ll_y_safe=cart_vert_list[2][1]-safe_dist
	cart_vert_ur_x_safe=cart_vert_list[3][0]-safe_dist
	cart_vert_ur_y_safe=cart_vert_list[3][1]+safe_dist
	sin_ang1 = (cart_vert_lr_y-cart_vert_ll_y)/len1
	sin_ang2 = (cart_vert_ur_y-cart_vert_ul_y)/len2
	ang1 = np.arcsin(sin_ang1)
	ang2 = np.arcsin(sin_ang2)
	ang_avg = (ang1+ang2)/2
	##print("Cart Average Angular Orientation: ", np.rad2deg(ang_avg))
	



	#pts1 = np.float32([[cart_vert_list[1][0],cart_vert_list[1][1]],
	#		[cart_vert_list[3][0],cart_vert_list[3][1]],
	#		[cart_vert_list[2][0],cart_vert_list[2][1]],
	#		[cart_vert_list[0][0],cart_vert_list[0][1]]])

	#pts2 = np.float32([[0,0],[len_avg,0],[0,wid_avg],[len_avg,wid_avg]])
	
	#mat_per = cv2.getPerspectiveTransform(pts1,pts2)

	#print("mat_per: ", mat_per)
	
	#frame_per = cv2.warpPerspective(frame_rot,mat_per,(int(len_avg),int(wid_avg)))

	# finding centroids of best_cnt and draw a circle there
	if isset('best_cnt'):
		#cv2.circle(frame,(int(cx),int(cy)),6,(0, 0, 255),-1)
		cv2.circle(frame,(int(cart_vert_list[0][0]),int(cart_vert_list[0][1])),3,(255,0, 0),-1)
		cv2.circle(frame,(int(cart_vert_list[1][0]),int(cart_vert_list[1][1])),3,(255,0, 0),-1)
		cv2.circle(frame,(int(cart_vert_list[2][0]),int(cart_vert_list[2][1])),3,(255,0, 0),-1)
		cv2.circle(frame,(int(cart_vert_list[3][0]),int(cart_vert_list[3][1])),3,(255,0, 0),-1)
		cv2.rectangle(frame, (cart_vert_ll_x_safe,cart_vert_ll_y_safe), (cart_vert_ur_x_safe,cart_vert_ur_y_safe), (255,0, 0), 2, 8, 0);
		#print("Central pos: (%d, %d)" % (cx,cy))
		#print("Central Rotated pos: (%d, %d)" % (mat_pts_rot[4][0],mat_pts_rot[4][1]))
	else:
		print("[Warning]Tag lost...")

	cart_len_ratio = (wid_avg/cart_width)
	cart_wid_ratio = (len_avg/cart_length)

	#print("cart_len_ratio: ", cart_len_ratio)
	#print("cart_wid_ratio: ", cart_wid_ratio)

	# Show the original and processed image
	cv2.imshow('frame', frame)
	#cv2.imshow('frame_per', frame_per)

	#cv2.imwrite('frame.png', frame);
	cv2.imwrite('thresh_cart.png', thresh_cart);

	#break



	#
	#
	# Setup color method for identifying N-objects larger than minimum threshold
 	#
	# This is for BROWN tag
	thresh_brown = cv2.inRange(hsv,np.array((10, 45, 70)), np.array((160, 190, 220)))

	# This is for BLACK tag
	thresh_black = cv2.inRange(hsv,np.array((105, 10, 12)), np.array((170, 160, 160)))

	thresh_combo = cv2.add(thresh_brown,thresh_black)
	thresh = cv2.subtract(thresh_combo,thresh_cart)

	cv2.imwrite('thresh_objects.png', thresh);

	thresh_1 = thresh.copy()

	# find contours in the threshold image
	im2, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


	# finding contour with maximum area and store it as best_cnt
	obj_dict = {0:"0"}
	obj_area_list = []
	obj_circle_contour_list = []
	obj_vert_list_overhung = []
	obj_vert_list_overhung_x0min = 99999
	obj_vert_list_overhung_x1max = -99999
	obj_vert_list_overhung_x2max = -99999
	obj_vert_list_overhung_x3min = 99999
	safe_vert_list = []
	detection_area_min = 500
	ind = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
		#print('approx',approx)
		if area > detection_area_min:
			if ((len(approx) > 50) & (area > 500) ):
				obj_circle_contour_list.append(cnt)
				obj_type = "circle"
			else:
				obj_type = "rectangle"
			obj_area_list.append(area)
			obj_dict[area] = [area, cnt, obj_type]
			ind = ind + 1
	obj_area_list.sort(reverse=True)
	obj_dict_ord = collections.OrderedDict(sorted(obj_dict.items()))
	obj_num = len(obj_area_list)
	#try: if circles is not None:
	obj_area_list_len = len(obj_area_list)
	if obj_area_list_len > 0:
		if max(obj_area_list) > detection_area_min:
			print("Number of Identified Objects Higher than Threshold: ", obj_num)
			for index in range(obj_num):
				##print("Detected Object[", cnt, "]: ", obj_dict_ord[obj_area_list[index]][0])
		
				M = cv2.moments(obj_dict_ord[obj_area_list[index]][1])
				cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
				cnt_center = [cx,cy]
				##print("Central Pos for Object: (%d, %d)" % (cx,cy))
			
				if obj_dict_ord[obj_area_list[index]][2] == 'circle':
					cv2.circle(frame,(cx,cy),100,(0,0,0),1)
					cv2.circle(frame, (x, y), r+10, (0, 255, 0), 4)
					#cv2.putText(frame,'Tire Detected',(10,20), font, 4,(0,255,0),2,cv2.LINE_AA)

					#Show the original and processed image
					cv2.imshow('frame', frame)
					cv2.imshow('thresh_load', thresh_1)

					cv2.imwrite('frame.png', frame);
					cv2.imwrite('thresh1.png', thresh_1);
				else:
					w, h = 2, 4;
					obj_vert_mat = [[0 for x in range(w)] for y in range(h)]

					max_pos_x = 0
					max_pos_y = 0
					min_pos_x = 99999
					min_pos_y = 99999
					for cnt in obj_dict_ord[obj_area_list[index]][1]:
						new_cnt=cnt[0]
						#print("new_cnt: ", new_cnt)
						contour_x = new_cnt[0]
						contour_y = new_cnt[1]
						if  (contour_x < cnt_center[0] and contour_y < cnt_center[1]):
							if (contour_x < min_pos_x and contour_y < min_pos_y):
								min_pos_x = contour_x
								#min_pos_y = contour_y
								#print("New max_pos_x: ", max_pos_x)
								obj_vert_mat[0][0] = contour_x
								obj_vert_mat[0][1] = contour_y
								if (contour_x < min_pos_x):
									min_pos_x = contour_x
									min_pos_y = contour_y
									#print("New max_pos_x: ", max_pos_x)
									obj_vert_mat[0][0] = contour_x
									obj_vert_mat[0][1] = contour_y
					max_pos_x = 0
					max_pos_y = 0
					min_pos_x = 99999
					min_pos_y = 99999
					for cnt in obj_dict_ord[obj_area_list[index]][1]:
						new_cnt=cnt[0]
						#print("new_cnt: ", new_cnt)
						contour_x = new_cnt[0]
						contour_y = new_cnt[1]
						if  (contour_x > cnt_center[0] and contour_y > cnt_center[1]):
							if (contour_x > max_pos_x and contour_y > max_pos_y):
								max_pos_x = contour_x				
								#max_pos_y = contour_y
								#print("New min_pos_x: ", min_pos_x)
								obj_vert_mat[1][0] = contour_x
								obj_vert_mat[1][1] = contour_y
								if (contour_x > max_pos_x):
									max_pos_y = contour_y
									max_pos_x = contour_x
									#print("New min_pos_x: ", min_pos_x)
									obj_vert_mat[1][0] = contour_x
									obj_vert_mat[1][1] = contour_y
					max_pos_x = 0
					max_pos_y = 0
					min_pos_x = 99999
					min_pos_y = 99999
					for cnt in obj_dict_ord[obj_area_list[index]][1]:
						new_cnt=cnt[0]
						#print("new_cnt: ", new_cnt)
						contour_x = new_cnt[0]
						contour_y = new_cnt[1]
						if  (contour_x > cnt_center[0] and contour_y < cnt_center[1]):
							if (contour_x > max_pos_x and contour_y < min_pos_y):		
								max_pos_x = contour_x			
								#min_pos_y = contour_y
								#print("New max_pos_y: ", max_pos_y)
								obj_vert_mat[2][0] = contour_x
								obj_vert_mat[2][1] = contour_y
								if (contour_x > max_pos_x):
									max_pos_x  = contour_x
									min_pos_y = contour_y
									#print("New min_pos_y: ", min_pos_y)
									obj_vert_mat[2][0] = contour_x
									obj_vert_mat[2][1] = contour_y
					max_pos_x = 0
					max_pos_y = 0
					min_pos_x = 99999
					min_pos_y = 99999
					for cnt in obj_dict_ord[obj_area_list[index]][1]:
						new_cnt=cnt[0]
						#print("new_cnt: ", new_cnt)
						contour_x = new_cnt[0]
						contour_y = new_cnt[1]
						if  (contour_x < cnt_center[0] and contour_y > cnt_center[1]):
							if (contour_x < min_pos_x and contour_y > max_pos_y):
								min_pos_x  = contour_x
								#max_pos_y = contour_y
								#print("New min_pos_y: ", min_pos_y)
								obj_vert_mat[3][0] = contour_x
								obj_vert_mat[3][1] = contour_y
								if (contour_x < min_pos_x):
									min_pos_x  = contour_x
									max_pos_y = contour_y
									#print("New min_pos_y: ", min_pos_y)
									obj_vert_mat[3][0] = contour_x
									obj_vert_mat[3][1] = contour_y

					##print("Objects Bounding Vertices: ", obj_vert_mat)

					if cart_vert_list[0][0] > obj_vert_mat[0][0]:
						if obj_vert_mat[0][0] < obj_vert_list_overhung_x0min:
							obj_vert_list_overhung.append([obj_vert_mat[0][0],obj_vert_mat[0][1]])
							obj_vert_list_overhung_x0min = obj_vert_mat[0][0]
					else:
						obj_vert_list_overhung.append([cart_vert_list[0][0],cart_vert_list[0][1]])
					if cart_vert_list[1][0] < obj_vert_mat[1][0]:
						if obj_vert_mat[0][0] > obj_vert_list_overhung_x1max:
							obj_vert_list_overhung.append([obj_vert_mat[1][0],obj_vert_mat[1][1]])
							obj_vert_list_overhung_x1max= obj_vert_mat[1][0]
					else:
						obj_vert_list_overhung.append([cart_vert_list[0][0],cart_vert_list[0][1]])
					if cart_vert_list[2][0] < obj_vert_mat[2][0]:
						if obj_vert_mat[0][0] > obj_vert_list_overhung_x2max:
							obj_vert_list_overhung.append([obj_vert_mat[2][0],obj_vert_mat[2][1]])
							obj_vert_list_overhung_x2max = obj_vert_mat[2][0]
					else:
						obj_vert_list_overhung.append([cart_vert_list[0][0],cart_vert_list[0][1]])
					if cart_vert_list[3][0] > obj_vert_mat[3][0]:
						if obj_vert_mat[0][0] < obj_vert_list_overhung_x3min:
							obj_vert_list_overhung.append([obj_vert_mat[3][0],obj_vert_mat[3][1]])
							obj_vert_list_overhung_x3min = obj_vert_mat[3][0]
					else:
						obj_vert_list_overhung.append([cart_vert_list[0][0],cart_vert_list[0][1]])
					obj_vert_lr_x=obj_vert_mat[0][0]
					obj_vert_lr_y=obj_vert_mat[0][1]
					obj_vert_ul_x=obj_vert_mat[1][0]
					obj_vert_ul_y=obj_vert_mat[1][1]
					obj_vert_ll_x=obj_vert_mat[2][0]
					obj_vert_ll_y=obj_vert_mat[2][1]
					obj_vert_ur_x=obj_vert_mat[3][0]
					obj_vert_ur_y=obj_vert_mat[3][1]

					#Geometric Analysis
					len1=((obj_vert_lr_x-obj_vert_ll_x)**2+(obj_vert_lr_y-obj_vert_ll_y)**2)**0.5
					len2=((obj_vert_ur_x-obj_vert_ul_x)**2+(obj_vert_ur_y-obj_vert_ul_y)**2)**0.5
					wid1=((obj_vert_ul_x-obj_vert_ll_x)**2+(obj_vert_ul_y-obj_vert_ll_y)**2)**0.5
					wid2=((obj_vert_ur_x-obj_vert_lr_x)**2+(obj_vert_ur_y-obj_vert_lr_y)**2)**0.5
					if (len1 == 0 and len2 == 0):
						len_avg = 99999
					else:
						if (len1 == 0):
							len_avg = len2
						if (len2 == 0):
							len_avg = len1
						if (len1 != 0 and len2 != 0):
							len_avg = (len1+len2)/2
					if (wid1 == 0 and wid2 == 0):
						wid_avg = 99999
					else:
						if (wid1 == 0):
							wid_avg = wid2
						if (wid2 == 0):
							wid_avg = wid1
						if (wid1 != 0 and wid2 != 0):
							wid_avg = (wid1+wid2)/2
					##print("Object Length: ", len_avg)
					##print("Object Width: ", wid_avg)

					sin_ang1 = (obj_vert_lr_y-obj_vert_ll_y)/len1
					sin_ang2 = (obj_vert_ur_y-obj_vert_ul_y)/len2
					ang1 = np.arcsin(sin_ang1)
					ang2 = np.arcsin(sin_ang2)
					ang_avg = (ang1+ang2)/2
					##print("Object Angle: ", np.rad2deg(ang_avg))

					# finding centroids of cnt2 (load object #1) and draw a circle there
					if isset('obj_dict_ord[obj_area_list[index]][1]'):
						cv2.circle(frame,(cx,cy),4,(0,0,0),-1)
						cv2.circle(frame,(obj_vert_mat[0][0],obj_vert_mat[0][1]),3,(0,0,0),-1)
						cv2.circle(frame,(obj_vert_mat[1][0],obj_vert_mat[1][1]),3,(0,0,0),-1)
						cv2.circle(frame,(obj_vert_mat[2][0],obj_vert_mat[2][1]),3,(0,0,0),-1)
						cv2.circle(frame,(obj_vert_mat[3][0],obj_vert_mat[3][1]),3,(0,0,0),-1)
					else:
						print("[Warning]Tag lost...")

					#Factor in Safety Zones Around Detected Objects
					obj_vert_lr_x_safe=obj_vert_mat[0][0]-safe_dist
					obj_vert_lr_y_safe=obj_vert_mat[0][1]-safe_dist
					obj_vert_ul_x_safe=obj_vert_mat[1][0]+safe_dist
					obj_vert_ul_y_safe=obj_vert_mat[1][1]+safe_dist
					obj_vert_ll_x_safe=obj_vert_mat[2][0]+safe_dist
					obj_vert_ll_y_safe=obj_vert_mat[2][1]-safe_dist
					obj_vert_ur_x_safe=obj_vert_mat[3][0]-safe_dist
					obj_vert_ur_y_safe=obj_vert_mat[3][1]+safe_dist
					sin_ang1 = (obj_vert_lr_y-obj_vert_ll_y)/len1
					sin_ang2 = (obj_vert_ur_y-obj_vert_ul_y)/len2
					ang1 = np.arcsin(sin_ang1)
					ang2 = np.arcsin(sin_ang2)
					ang_avg = (ang1+ang2)/2

					# finding centroids of best_cnt and draw a circle there
					if isset('obj_dict_ord[obj_area_list[index]][1]'):
						cv2.circle(frame,(obj_vert_lr_x_safe,obj_vert_lr_y_safe),3,(0,0,0),-1)
						cv2.circle(frame,(obj_vert_ul_x_safe,obj_vert_ul_y_safe),3,(0,0,0),-1)
						cv2.circle(frame,(obj_vert_ll_x_safe,obj_vert_ll_y_safe),3,(0,0,0),-1)
						cv2.circle(frame,(obj_vert_ur_x_safe,obj_vert_ur_y_safe),3,(0,0,0),-1)
						cv2.rectangle(frame, (obj_vert_ll_x_safe,obj_vert_ll_y_safe), (obj_vert_ur_x_safe,obj_vert_ur_y_safe), (0, 0, 0), 2, 8, 0);
						#cv2.putText(frame,'Object Detected',(10,10), font, 4,(0,255,0),2,cv2.LINE_AA)

				#break
		else:
			print("NC Cart is Empty")
			print("Please Return to Load Station")
	#except:
			#print("NC Cart is Empty")
			#print("Please Return to Load Station")


	#Show the original and processed image
	if len(obj_vert_list_overhung) > 0:
		obj_vert_list_overhung.sort(reverse=True)
		print('obj_vert_list_overhung reverse: ', obj_vert_list_overhung)
		cart_vert_list.sort(reverse=True)
		print('cart vertices reverse: ', cart_vert_list)
		if cart_vert_list[0][0] < obj_vert_list_overhung[0][0]:
			safe_vert_list.append([obj_vert_list_overhung[0][0]+safe_dist,cart_vert_list[0][1]+safe_dist])
		else:
			safe_vert_list.append([cart_vert_list[0][0]+safe_dist,cart_vert_list[0][1]+safe_dist])

		obj_vert_list_overhung.sort()
		print('obj_vert_list_overhung: ', obj_vert_list_overhung)
		cart_vert_list.sort()
		print('cart vertices: ', cart_vert_list)
		if cart_vert_list[0][0] > obj_vert_list_overhung[0][0]:
			safe_vert_list.append([obj_vert_list_overhung[0][0]-safe_dist,cart_vert_list[0][1]-safe_dist])
		else:
			safe_vert_list.append([cart_vert_list[0][0]-safe_dist,cart_vert_list[0][1]-safe_dist])

		obj_vert_list_overhung.sort(key=lambda x: x[1])
		print('obj_vert_list_overhung: ', obj_vert_list_overhung)
		cart_vert_list.sort(key=lambda x: x[1])
		print('cart vertices: ', cart_vert_list)
		if cart_vert_list[0][1] > obj_vert_list_overhung[0][1]:
			safe_vert_list.append([obj_vert_list_overhung[0][0]-safe_dist,cart_vert_list[0][1]-safe_dist])
		else:
			safe_vert_list.append([cart_vert_list[0][0]-safe_dist,cart_vert_list[0][1]-safe_dist])

		obj_vert_list_overhung.sort(reverse=True, key=lambda x: x[1])
		print('obj_vert_list_overhung: ', obj_vert_list_overhung)
		cart_vert_list.sort(reverse=True, key=lambda x: x[1])
		print('cart vertices: ', cart_vert_list)
		if cart_vert_list[0][1] < obj_vert_list_overhung[0][1]:
			safe_vert_list.append([obj_vert_list_overhung[0][0]+safe_dist,cart_vert_list[0][1]+safe_dist])
		else:
			safe_vert_list.append([cart_vert_list[0][0]+safe_dist,cart_vert_list[0][1]+safe_dist])

		safe_vert_list.sort()
		print('safe_vert_list: ', safe_vert_list)
		cv2.rectangle(frame, (safe_vert_list[1][0],safe_vert_list[1][1]), (safe_vert_list[3][0],safe_vert_list[3][1]), (0, 0, 255), 2, 8, 0);

	cv2.imshow('frame', frame)
	cv2.imshow('thresh_load', thresh)

	cv2.imwrite('frame.png', frame);

	img_index = img_index + 1

	# if key pressed is 'Esc' then exit the loop
	if cv2.waitKey(33)== 27:
		break

# Clean up and exit the program
cv2.destroyAllWindows()
cap.release()
