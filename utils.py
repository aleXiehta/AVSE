import sys
import os
import glob
import argparse

import numpy as np
# import cv2
import dlib

from skimage import io

class FacePreprocessor(object):
    
    def __init__(self, predictor_path):
        self.predictor_path = predictor_path

    def get_bounding_box(self, landmarks, square=False):
        """
        Args:
            landmarks (list): List of numpy ndarrays of facial landmarks.
            square (bool): Restrict bounding box to square, default as "False".

        Returns:
            rmin (int): Minimum row index.
            rmax (int): Maximum row index.
            cmin (int): Minimum column index.
            cmax (int): Maximum column index.

        """
        rmin = landmarks[:, 1].min()
        rmax = landmarks[:, 1].max()
        cmin = landmarks[:, 0].min()
        cmax = landmarks[:, 0].max()

        if square:
            if rmax - rmin > cmax - cmin:
                diff = (rmax - rmin) // 2
                center = (cmax + cmin) // 2
                cmin, cmax = center - diff, center + diff

            else:
                diff = (cmax - cmin) // 2
                center = (rmax + rmin) // 2
                rmin, rmax = center - diff, center + diff

        return rmin, rmax, cmin, cmax

    def face_detector(self, f_path, return_part='mouth', multi_face=False, square=True):
        """
        Args:
            f_path (str): File path of face image.

            p_path (str): File path of face landmark predictor.

            return_part (str):
                The part to be returned.
                Including:
                    [1] jaw
                    [2] leyebrow
                    [3] reyebrow
                    [4] nose
                    [5] leye
                    [6] reye
                    [7] mouth

            multi_face (bool):
                True: Return landmarks of all face.
                False: Return landmarks of largest selected return part.

            square (bool): Restrict the bounding box to square, default as "False".

        Returns:
            landmarks (list): List of numpy ndarray of landmarks.
        """
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.predictor_path)

        img = io.imread(f_path)
        dets = detector(img)

        shapes = []
        coordinates = []

        for k, d, in enumerate(dets):
            shapes.append(predictor(img, d))
            coordinates.append(np.empty([68, 2], dtype=int)) # landmark coordinates

        # face landmarks:
        #  0 - 16: jaw
        # 17 - 21: left eyebrow
        # 22 - 26: right eyebrow
        # 27 - 35: nose
        # 36 - 41: left eye
        # 42 - 47: right eye
        # 48 - 67: mouth

        for k, shape in enumerate(shapes):
            for i in range(68):
                try:
                    coordinates[k][i][0] = shape.part(i).x
                    coordinates[k][i][1] = shape.part(i).y
                
                except:
                    print('Face cannot be found in frame: "{}".'.format(f_path))
                    break


        all_landmarks = []

        for k, coordinate in enumerate(coordinates):
            # print("Face #{}".format(k))
            # print("Coordinate: {}".format(coordinate[k]))
            landmarks = dict()
            landmarks['jaw'] = coordinate[0:17]
            landmarks['leyebrow'] = coordinate[17:22]
            landmarks['reyebrow'] = coordinate[22:27]
            landmarks['nose'] = coordinate[27:36]
            landmarks['leye'] = coordinate[36:42]
            landmarks['reye'] = coordinate[42:48]
            landmarks['mouth'] = coordinate[48:68]
            landmarks['all'] = coordinate

            all_landmarks.append(landmarks[return_part])

        if multi_face:
            return all_landmarks

        # Find the largest face organ
        else:
            cropped_area = []
            for landmarks in all_landmarks:
                # print(landmarks)
                rmin, rmax, cmin, cmax = self.get_bounding_box(
                        landmarks, 
                        square=square)

                cropped_area.append((rmax - rmin) * (cmax - cmin))
            
            cropped_area = np.array(cropped_area)
            index = cropped_area.argmax()

            return [all_landmarks[index]]

    def crop(self, img, landmarks, r_ext=0, c_ext=0, square=True):
        rmin, rmax, cmin, cmax = self.get_bounding_box(landmarks, square)
        cropped = img[rmin - r_ext :rmax + r_ext, cmin - c_ext:cmax + c_ext]

        return cropped

    def gen_cropped_mouth(self, face_root, output_path):
        # dir structure: <speaker_utterance>/video/<frames>.png
        speaker_paths = sorted(glob.glob(os.path.join(face_root, '*')))
        
        # generate lips paths
        for s_path in speaker_paths:
            os.makedirs(os.path.join(s_path, 'lip'), exist_ok=True)
            frame_paths = sorted(glob.glob(os.path.join(s_path, 'video', '*.png')))
            for f_path in frame_paths:
                print('Extracting mouth from "{}"...'.format(f_path), end='\r')
                landmarks = self.face_detector(f_path, multi_face=False)

                for i, landmark in enumerate(landmarks):
                    cropped = self.crop(io.imread(f_path), landmark, square=True)
                    io.imsave(f_path.replace('video', 'lip'), cropped)


class VideoPreprocessor(object):
    
    def __init__(self):
        pass

    def get_video_paths(self, video_root):
        '''
        Args:
            video_root (str): Root path to videos.

        Returns:
            video_paths (str): Paths of videos.            
        '''
        video_paths = []
        for root, dirs, files in os.walk(video_root):
            for f in files:
                video_paths.append(os.path.join(root, f))

        return video_paths

    def extract(self, video_root, output_path, fr=50, sr=16000):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # list all video
        video_paths = self.get_video_paths(video_root)
        num_video_paths = len(video_paths)
        print('Number of videos: {}'.format(num_video_paths))

        for i, v_path in enumerate(video_paths):
            try:
                # Display progress
                print('[{}/{}]Extracting audio/frame from "{}".'.format(
                    i + 1,
                    num_video_paths,
                    v_path))

                f_path = v_path.split('/')
                f_path = f_path[-1]

                speaker = f_path[:f_path.rfind('.')]

                f_path = os.path.join(output_path, f_path)

                # f_path = os.path.join(*f_path)
                folder_name = f_path[:f_path.rfind('.')]

                v_folder_name = os.path.join(folder_name, 'video')
                a_folder_name = os.path.join(folder_name, 'audio/clean')

                # Create audio/video folders
                os.makedirs(v_folder_name, exist_ok=True)
                os.makedirs(a_folder_name, exist_ok=True)
                
                # Extract video
                os.system(\
                'ffmpeg -loglevel panic -i {} -vf fps={} {}_%3d.png -hide_banner'.format(
                    v_path, 
                    fr, 
                    os.path.join(v_folder_name, speaker)))

                # Extract audio
                os.system(\
                'ffmpeg -loglevel panic -i {} -ab 1k -ac 1 -ar {} -vn {}.wav'.format(
                    v_path, 
                    sr,
                    os.path.join(a_folder_name, speaker)))

            except:
                print('Extraction failed on "{}".'.format(v_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor-path', type=str)
    parser.add_argument('--face-path', type=str)
    parser.add_argument(
            '--output-path', type=str, help='Ouput path for cropped mouth images')
    args = parser.parse_args()

    f = FacePreprocessor(args.predictor_path)

    # test face detector
    landmarks = f.face_detector(args.face_path, multi_face=True)

    img = io.imread(args.face_path)

    for i, landmark in enumerate(landmarks):
        cropped = f.crop(img, landmark, square=True)
        io.imsave('out_{}.png'.format(i), cropped)
