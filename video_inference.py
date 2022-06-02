import os
from config import cfg
import cv2
from image_inference import InferenceYOLOV5, ModelFileYOLOv5


class Video:
    def __init__(self, target_gpu_id, model_files, overlay_detections=False):
        self.image_infer = InferenceYOLOV5(target_gpu_id, model_files, overlay_detections)

    def main(self, source):
        output_vid = f'{cfg.video.output_folder}{source.split("/")[-1].split(".")[0]}_output.mp4'
        writer = None
        vs = cv2.VideoCapture(source)
        while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break
            try:
                bboxes, scores, names, class_ids, img_out = self.image_infer.inferYOLOV5(frame)

                if cfg.flags.image_show:
                    cv2.imshow("Frame", img_out)
                    key = cv2.waitKey(1) & 0xFF
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break

                if cfg.flags.video_write and writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*cfg.video.FOURCC)
                    writer = cv2.VideoWriter(output_vid, fourcc, cfg.video.video_writer_fps,
                                             (img_out.shape[1], img_out.shape[0]), True)
                # if the video writer is not None, write the frame to the output
                # video file
                if writer is not None:
                    writer.write(img_out)
            except TypeError:
                pass


if __name__ == '__main__':
    # initialized the class
    model_directory = './model_data'
    model_files = ModelFileYOLOv5(model_directory)
    target_gpu_device = '0'
    detector = Video(target_gpu_device, model_files, overlay_detections=True)

    # set your path to video
    video_source = ''
    if not os.path.exists(video_source):
        print('Video {} does not exists'.format(video_source))
    # make output_vid = '' if you don't want to save the video
    detector.main(video_source)
