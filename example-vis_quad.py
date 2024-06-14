# Copyright Oliver Zendel, AIT Austrian Institute of Technology Gmbh 2019
# Example visualization tool to illustrate interpretation of RailSem19 data usage

import sys, os, glob,  math, argparse
import numpy as np
import cv2, json

rs19_label2bgr = {"buffer-stop": (70,70,70),
                  "crossing": (128,64,128),
                  "guard-rail": (0,255,0),
                  "train-car" :  (100,80,0),
                  "platform" : (232,35,244),
                  "rail": (255,255,0),
                  "switch-indicator": (127,255,0),
                  "switch-left": (255,255,0),
                  "switch-right": (127,127,0),
                  "switch-unknown": (191,191,0),
                  "switch-static": (0,255,127),
                  "track-sign-front" : (0,220,220),
                  "track-signal-front" : (30,170,250),
                  "track-signal-back" : (0,85,125),
                  #rail occluders
                  "person-group" : (60,20,220),
                  "car" : (142,0,0),
                  "fence" : (153,153,190),
                  "person" : (60,20,220),
                  "pole" : (153,153,153),
                  "rail-occluder" : (255,255,255),
                  "truck" : (70,0,0)
                }

def files_in_subdirs(start_dir, pattern=["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend((sorted(glob.glob(os.path.join(dir,p)))))
    return files

def config_to_rgb(inp_path_config_json, default_col=[255,255,255]):
    lut = []
    inp_json = json.load(open(inp_path_config_json, 'r'))
    
    for c in range(3): #for each color channel
        lut_c =[l["color"][c] for l in inp_json["labels"]]+[default_col[c]]*(256-len(inp_json["labels"]))
        lut.append(np.asarray(lut_c, dtype=np.uint8))

    return lut


def corss_hatch_rail(im_vis, coords, color_left=(255,255,0), color_right=(127,127,0)):
    ml = min(len(coords[0]), len(coords[1]))
    for i in range(ml):
        midpnt = ((coords[0][i][0]+coords[1][i][0])//2, (coords[0][i][1]+coords[1][i][1])//2)
        cv2.line(im_vis, tuple(coords[0][i]), midpnt, color_left)
        cv2.line(im_vis, midpnt, tuple(coords[1][i]), color_right)
        

def json_to_img(inp_path_json, line_thickness=3):
    inp_json = json.load(open(inp_path_json, 'r'))
    im_json = np.zeros((inp_json["imgHeight"], inp_json["imgWidth"], 3), dtype=np.uint8)

    image = cv2.imread(f'{inp_path_json.replace("json", "jpg")}')
    label_list = []
    
    for obj in inp_json["objects"]:
        col = rs19_label2bgr.get(obj["label"],[255,255,255])
        label = obj.get('label')

        if "boundingbox" in obj:
            cv2.rectangle(im_json, tuple(obj["boundingbox"][0:2]), tuple(obj["boundingbox"][2:4]), col, line_thickness)
            
            cv2.rectangle(image, tuple(obj["boundingbox"][0:2]), tuple(obj["boundingbox"][2:4]), (0, 0, 0), line_thickness)
            label_list.append([image, label, (obj["boundingbox"][0], obj["boundingbox"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), line_thickness])
            
            print(f'Bounding box: {label}')

        elif "polygon" in obj:
            pnts_draw = np.around(np.array(obj["polygon"])).astype(np.int32)
            cv2.polylines(im_json, [pnts_draw], True, col, line_thickness)

            cv2.polylines(image, [pnts_draw], True, (0, 0, 255), line_thickness)            
            label_list.append([image, label, (pnts_draw[0][0], pnts_draw[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), line_thickness])
            
            print(f'Polygon: {label}')

        elif "polyline-pair" in obj:
            #left rail of a rail pair has index 0, right rail of a rail pair has index 1
            rails_draw = [np.around(np.array(obj["polyline-pair"][i])).astype(np.int32) for i in range(2)]
            corss_hatch_rail(im_json, obj["polyline-pair"],  rs19_label2bgr['switch-left'], rs19_label2bgr['switch-right'])
            cv2.polylines(im_json, rails_draw, False, col)

            corss_hatch_rail(image, obj["polyline-pair"],  rs19_label2bgr['switch-left'], rs19_label2bgr['switch-right'])
            cv2.polylines(image, rails_draw, False, (255, 0, 0), line_thickness)            
            label_list.append([image, label, (rails_draw[0][0][0], rails_draw[0][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), line_thickness])
            
            print(f'Polyline-pair: {label}')

        elif "polyline" in obj:
            rail_draw = np.around(np.array(obj["polyline"])).astype(np.int32)
            cv2.polylines(im_json, [rail_draw], False, col, line_thickness)

            cv2.polylines(image, [rail_draw], False, (0, 255, 0), line_thickness)
            label_list.append([image, label, (rail_draw[0][0], rail_draw[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), line_thickness])
            
            print(f'Polyline: {label}')

        for arg in label_list:
            cv2.putText(*arg)
            
    return im_json, inp_json["frame"], image
     
def process_image(image):
    new_width = image.shape[1] // 2
    new_height = image.shape[0] // 2
    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image
     
def get_joined_img(inp_path_json, jpg_folder, uint8_folder, lut_bgr, blend_vals=[0.65,0.25,0.1]):
    im_json, frameid, geo_image = json_to_img(inp_path_json, line_thickness=2) #visualize geometric annotations
    inp_path_jpg = os.path.join(jpg_folder, frameid+".jpg")  
    inp_path_uint8 = os.path.join(uint8_folder, frameid+".png")
    im_jpg = cv2.imread(inp_path_jpg) #add intensity image as background
    im_id_map = cv2.imread(inp_path_uint8,0) #get semantic label map
    print(im_id_map.shape)
    im_id_col = np.zeros((im_id_map.shape[0], im_id_map.shape[1], 3), np.uint8)
    for c in range(3):
        im_id_col[:,:,c] = lut_bgr[c][im_id_map] #apply color coding       
        
    origin_image = (im_jpg).astype(np.uint8)
    mixed_image = (im_jpg*blend_vals[0]+im_id_col*blend_vals[1]+im_json*blend_vals[2]).astype(np.uint8)
    smt_image = (im_id_col).astype(np.uint8)
    
    cv2.imwrite(f'./res/test_vis.jpg', smt_image)
    
    image_list = [origin_image, geo_image, smt_image, mixed_image]
    
    return list(map(process_image, image_list)) #blend all three data sources
    # return (im_id_col).astype(np.uint8), image #blend all three data sources

     
def vis_all_json(json_folder, jpg_folder, uint8_folder, inp_path_config_json):
    all_json = files_in_subdirs(json_folder, pattern = ["*.json"])
    lut_bgr = config_to_rgb(inp_path_config_json, default_col = [255,255,255])[::-1] #we need to swap color channels as opencv uses BGR
    
    curr_idx, retKey = 0, ord('a')
    
    while retKey != 27:
        print(f'\n{all_json[curr_idx]}')
        image_list = get_joined_img(all_json[curr_idx], jpg_folder, uint8_folder, lut_bgr)
        cv2.putText(image_list[0], all_json[curr_idx],(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        
        for img in image_list:
            img = cv2.resize(img, (1920 // 2, 1080 // 2))
        img_u, img_d = np.hstack((image_list[0], image_list[1])), np.hstack((image_list[2], image_list[3]))
        img_tot = np.vstack((img_u, img_d))    
            
        cv2.imshow("RailSem19 Annotation Visualization (use 'a' or 'd' to navigate, ESC to quit)", img_tot)
        retKey = cv2.waitKey(-1) #use 'a' and 'd' to scroll through frames
        if retKey == ord('a'):
            curr_idx -= 1
        elif retKey == ord('d'):
            curr_idx += 1
        elif retKey == ord('\r') or retKey == ord('\n'):            
            input_idx = int(input("Enter desired index: "))
            if input_idx >= 0 and input_idx < len(all_json):
                curr_idx = input_idx
            else:
                print(f"Index {input_idx} out of range. Please enter a valid index.")

        curr_idx = max(0, min(curr_idx, len(all_json) - 1))  # Ensure curr_idx stays within valid range
              
    return 0


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpgs', type=str, default="./jpgs/rs19_val",
                        help="Folder containing all RGB intensity images")
    parser.add_argument('--jsons', type=str, default="./jsons/rs19_val",
                        help="Folder containing all geometry-based annotation files")
    parser.add_argument('--uint8', type=str, default="./uint8/rs19_val",
                        help="Folder containing all dense semantic segm. id maps")
    parser.add_argument('--config', type=str, default="./rs19-config.json",
                        help="Path to config json for dense label map interpretation")
    args = parser.parse_args(argv)
    return vis_all_json(args.jsons, args.jpgs, args.uint8, args.config)
    
if __name__ == "__main__":
    sys.exit(main())
