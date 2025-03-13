import cv2
import numpy as np
import subprocess
from pathlib import Path
import os
import logging
from pathlib import Path
from typing import Union, Optional

def convert_mp4_to_mkv(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    ffmpeg_path: str = "ffmpeg",
    overwrite: bool = False
) -> bool:
    """
    Convert an MP4 file to MKV format using ffmpeg.
    
    Args:
        input_path: Path to the input MP4 file
        output_path: Path for the output MKV file. If None, uses the same path as input with .mkv extension
        ffmpeg_path: Path to ffmpeg executable
        overwrite: If True, overwrites existing output file
    
    Returns:
        bool: True if conversion was successful, False otherwise
    
    Raises:
        FileNotFoundError: If input file doesn't exist or ffmpeg is not found
        ValueError: If input file is not an MP4
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Convert input_path to Path object
    input_path = Path(input_path)
    
    # Validate input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_path.suffix.lower() != '.mp4':
        raise ValueError(f"Input file must be an MP4 file, got: {input_path.suffix}")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('.mkv')
    else:
        output_path = Path(output_path)
    
    # Check if output file already exists
    if output_path.exists() and not overwrite:
        logger.error(f"Output file already exists: {output_path}")
        return False
    
    # Prepare ffmpeg command
    command = [
        ffmpeg_path,
        '-i', str(input_path),
        '-c', 'copy',  # Copy streams without re-encoding
        '-y' if overwrite else '-n',  # Overwrite output if specified
        str(output_path)
    ]
    
    try:
        # Run ffmpeg command
        logger.info(f"Converting {input_path} to {output_path}")
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if conversion was successful
        if process.returncode == 0:
            logger.info("Conversion completed successfully")
            return True
        else:
            logger.error(f"Conversion failed with error: {process.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e}")
        return False
    except FileNotFoundError:
        logger.error(f"FFmpeg not found at: {ffmpeg_path}")
        raise FileNotFoundError("FFmpeg not found. Please ensure it's installed and accessible")

def detect_scene(video_path, frame_rate=5):
    """
    Detect scene changes in a video by analyzing frame differences.
    
    Args:
        video_path (str): Path to the input video file
        
    Returns:
        list: Timestamps (in seconds) where significant scene changes occur
    """

    # Create the folde
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * fps)  # Number of frames for a 5-second interval
    frame_count = 0
    
    # Initialize first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video")
        cap.release()
        return []
    
    x1, y1 = 50, 150# Top-left corner
    x2, y2 = 4000, 4000  # Bottom-right corner
    
    timestamp_start = []
    
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += frame_interval
        
        frame_count += frame_interval  # Skip to the next 10-second mark
        cropped_frame = frame[y1:y2, x1:x2]

        hsv_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])   # Lower range for red
        upper_red1 = np.array([10, 255, 255]) # Upper range for red
        lower_red2 = np.array([170, 120, 70]) # Another lower range for red (due to HSV wrap-around)
        upper_red2 = np.array([180, 255, 255]) # Upper range for red
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        # Combine both masks
        red_mask = mask1 + mask2

        # Check if red is present
        if red_mask.sum() > 400000:
            timestamp = frame_count / fps
            timestamp_start.append(timestamp)
    
    cap.release()
    return timestamp_start

def extract_range(numbers):
    """
    Group timestamps into ranges based on proximity.
    
    Args:
        numbers (list): List of timestamps
        
    Returns:
        list: List of tuples containing start and end times for each range
    """
    if not numbers:
        return []
    
    result = []
    current_range = []
    
    for i in range(len(numbers)):
        if i == 0:
            current_range = [numbers[i]]
        elif i == len(numbers) - 1:
            current_range.append(numbers[i])
            result.append((current_range[0], current_range[-1]))
        elif numbers[i+1] - numbers[i] > 100:  # If gap is too large, start new range
            current_range.append(numbers[i])
            result.append((current_range[0], current_range[-1]))
            current_range = []
        else:
            current_range.append(numbers[i])
    
    return result

def trim_video(input_path: str, output_path: str, start_time: str, end_time: str) -> None:
    """
    Trim a video using FFmpeg stream copying (no re-encoding).
    
    Args:
        input_path: Path to input video file
        output_path: Path to save trimmed video
        start_time: Start timestamp in format "HH:MM:SS"
        end_time: End timestamp in format "HH:MM:SS"
    """
    try:
        command = [
            'ffmpeg',
            '-i', input_path,
            '-ss', start_time,
            '-to', end_time,
            '-c', 'copy',
            '-map', '0',
            output_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully trimmed video to: {output_path}")
        else:
            print(f"Error trimming video: {result.stderr}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def seconds_to_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Main execution
if __name__ == "__main__":
    # Input configuration
    video_path = 'videos\A02_20241002101424.mp4'  # Change this to your video path
    video_format = video_path.split('.')[-1]

    if video_format == 'mp4':

        temp_path = video_path.split('.')[0]
        temp_video_path = temp_path + '.mkv'

        convert_mp4_to_mkv(video_path, temp_video_path)

        video_path = temp_video_path


    padding_seconds = 20  # Number of seconds to add before and after each scene
    
    # Ensure input video exists
    if not Path(video_path).exists():
        print(f"Error: Input video file '{video_path}' does not exist.")
        exit()
    
    print(f"Processing video: {video_path}")

    # Extract folder name by removing the file extension
    folder_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(folder_name, exist_ok=True)
    report_file_path = f'{folder_name}/report.txt'
    
    # Detect scenes
    detected_timestamps = detect_scene(video_path)
    if not detected_timestamps:
        print("No scenes detected in the video.")
        
        with open(report_file_path, 'a') as f:

                f.write("No scenes detected in the video.")

        exit()

    
    
    # Group timestamps into ranges
    timestamp_ranges = extract_range(detected_timestamps)
    

    
    report_file_path = f'{folder_name}/report.txt'

    with open(report_file_path, 'a') as f:

            f.write(f"Number of Scene in the video: {len(timestamp_ranges)}:")
    # Process each range
    for i, (start, end) in enumerate(timestamp_ranges):
        # Add padding to start and end times
        padded_start = max(0, start - padding_seconds)
        padded_end = end + padding_seconds
        
        # Convert to timestamp format
        start_timestamp = seconds_to_timestamp(padded_start)
        end_timestamp = seconds_to_timestamp(padded_end)
        
        # Create output filename
        output_path = f"{folder_name}/scene_{i+1}.mkv"
    
        with open(report_file_path, 'a') as f:

            f.write(f"\nscene {i+1}:")
            f.write(f"Time range: {start_timestamp} to {end_timestamp}")
        
        # Trim video
        trim_video(video_path, output_path, start_timestamp, end_timestamp)