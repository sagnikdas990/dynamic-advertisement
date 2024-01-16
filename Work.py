import cv2
import numpy as np
import moviepy.editor as mp
#from google.colab.patches import cv2_imshow


def read_video(video_path):
    return cv2.VideoCapture(video_path)


def segment_objects(frame):
    # Perform object segmentation using GrabCut algorithm
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, frame.shape[1] - 50, frame.shape[0] - 50)  # Region of interest (ROI)
    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create segmentation mask
    segmentation_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return segmentation_mask

def add_logo(im_dst, logo,):
    pts_dst = np.array([[593,223],[983,223], [1020, 325], [550, 325]]) 
    
    h, w, c = logo.shape

    pts_src = np.array([[0, 0],[w-1, 0],[w-1, h-1],[0, h-1]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_temp = cv2.warpPerspective(logo, h, (im_dst.shape[1],im_dst.shape[0]))

    cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0);
    
    # Add warped source image to destination image.
    im_dst = im_dst + im_temp;
    
    return im_dst


def replace_random_area(frame, replacement_image, pm,gm):
    # Choose a random area on the ground and replace it with another image
    # If a player enters the area, the player will be visible on the new image

    # Generate random coordinates for the area
    ground_area_height, ground_area_width, _ = replacement_image.shape
    y = np.random.randint(0, frame.shape[0] - ground_area_height)
    x = np.random.randint(0, frame.shape[1] - ground_area_width)

    # Resize the replacement image to match the size of the ground area
    resized_replacement_image = cv2.resize(replacement_image, (ground_area_width, ground_area_height))

    # Replace the area with the resized replacement image
    frame[y:y+ground_area_height, x:x+ground_area_width] = resized_replacement_image

    return frame

def overlay_logo(frame, replacement_image, mask,mask_inv):
    #frame = add_logo(frame, replacement_image)

    mask = cv2.merge([mask,mask,mask])
    mask_inv = cv2.merge([mask_inv,mask_inv,mask_inv])

    masked = frame*(mask/255)

    after_logo_add = add_logo(frame, replacement_image)
    masked_inv = after_logo_add*(mask_inv/255)
    # out = add_logo(masked_inv*255, logo, pts_dst)
    # out = out/255
    final = (masked+ masked_inv).astype(np.uint8)
    
    return final



def main():
    video_path = "1.mp4"
    replacement_image_path = "logo.png"
    output_video_path = "output_video.mp4"
     # Set the maximum number of frames to process

    cap = read_video(video_path)
    replacement_image = cv2.imread(replacement_image_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, codec, fps, (frame_width, frame_height))

    video = mp.VideoFileClip(video_path)
    logo = (mp.ImageClip(replacement_image_path)
            .set_duration(video.duration)
            .resize(height=50)
            .margin(right=20, top=20, opacity=0)
            .set_pos('right', 'top'))

    final = mp.CompositeVideoClip([video, logo])

    frame_count = 0
    while frame_count < 250:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 2: Detect players and football
        segmentation_mask = segment_objects(frame)

        player_mask = np.zeros_like(segmentation_mask)
        player_mask[segmentation_mask == 1] = 255  # Assuming class 1 corresponds to players

        football_mask = np.zeros_like(segmentation_mask)
        football_mask[segmentation_mask == 2] = 255  # Assuming class 2 corresponds to the football

        # Step 3: Replace a random area on the ground with the replacement image
        ground_area_mask = ~player_mask & ~football_mask

        if np.any(ground_area_mask):  # If there is a valid ground area, replace it
            frame = overlay_logo(frame, replacement_image, ground_area_mask, player_mask)

        out.write(frame)

        cv2.imshow(' - ',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    main()